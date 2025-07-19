import networkx as nx
import geopandas as gpd
from shapely import wkb
import binascii
from sqlalchemy import create_engine
import pandas as pd
from charging_stations.refactored_charging_station import ChargingStation, Port
import pickle
import random

# 1. Connect ke PostGIS
engine = create_engine("postgresql://o2p:o2p@localhost:5433/o2p")

# 2. Load edges (snapped_lines)
edges = gpd.read_postgis("""
    SELECT osm_id, source, target, way AS geometry, ST_Length(way) AS length, amenity, highway
    FROM snapped_lines
""", engine, geom_col='geometry')

# 3. Load nodes (snapped_lines_vertices_pgr)
nodes = gpd.read_postgis("""
    SELECT id, the_geom AS geometry, cs_id
    FROM snapped_lines_vertices_pgr
""", engine, geom_col='geometry')

# 4. Buat directed graph
G = nx.MultiDiGraph()

nodes = nodes[(nodes['geometry'].geom_type != 'Point') | (~nodes['geometry'].is_empty)]

# 5. Tambahkan simpul (nodes)
for idx, row in nodes.iterrows():
    G.add_node(row['id'], x=row.geometry.x, y=row.geometry.y, geometry=row.geometry, cs_id=row.cs_id)

# 6. Tambahkan edge
for idx, row in edges.iterrows():
    G.add_edge(
        row['source'], row['target'],
        id=row['osm_id'],
        key=row['osm_id'],
        length=row['length'],
        geometry=row['geometry'],
        amenity=row['amenity'],
        highway=row['highway']
    )

G.remove_nodes_from([
    n for n, d in G.nodes(data=True)
    if 'geometry' not in d or d['geometry'].is_empty or 'x' not in d or 'y' not in d
])

sql = """
SELECT *
FROM charging_station
"""

df = pd.read_sql(sql, engine)

query_ports = "SELECT * FROM charging_port"
df_ports = pd.read_sql(query_ports, engine)

ports_by_cs = {}

for cs_id, group in df_ports.groupby('cs_id'):
    ports_list = []

    for _, row in group.iterrows():
        port_obj = Port(
            env=None,
            id=row['number'],
            portType=row['connector_type'],
            power=row['power'],
            isAvailable=True,
            price=row["price"],
            portSession=None
        )
        ports_list.append(port_obj)
    ports_by_cs[cs_id] = ports_list

cs_dict = df.set_index('id').to_dict(orient='index')

facilities = pd.read_excel('./resources/Sarana.xlsx')
facilities.loc[facilities['Nama Sarana'].isin(['Masjis','WC Umum'])]

facilities_list = facilities['Nama Sarana'].loc[facilities['Nama Sarana'].isin(['WC Umum','Masjid'])]
facilities_list = facilities_list.to_list()

for node, data in G.nodes(data=True):
    cs_id = data.get('cs_id')
    if cs_id is not None:
        cs_info = cs_dict.get(cs_id)
        if cs_info is not None:
            geom = wkb.loads(binascii.unhexlify(cs_info.get('the_geom')))
            lon, lat = geom.x, geom.y
            cs_obj = ChargingStation(
                env=None,
                id=cs_id,
                name=cs_info.get('name'),
                ports=ports_by_cs.get(cs_id, None),
                lat=lat,
                lon=lon,
                simulation=None,
                node_id=node,
                facilities=random.choice([facilities_list])
            )
            if cs_obj.ports == None:
                continue
            G.nodes[node]['charging_station'] = cs_obj
        else:
            print(f"cs_id {cs_id} not found in cs_dict")
    else:
        print(f"Node {node} does not have cs_id")

with open("resources/graph_with_cs.pkl", "wb") as f:
    pickle.dump(G, f)