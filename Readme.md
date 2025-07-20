# EV Recharging Scheduling System Documentation


## Early Setup

Clone this repository

```bash
git clone https://github.com/buatanalan/EV-Scheduling
```

Create a virtual environment
```bash
python -m venv .venv
```

Activate virtual environment
```bash
source .venv/bin/activate
```

Install necessary libarary
```bash
pip install -r requirements.txt
```

### Docker preparation

1. Turn on docker
2. Open Terminal
3. Install Redis
```bash
docker run -d --name redis -p 6379:6379 redis:7.2
```
4. Install message broker eqmx
```bash
docker run -d --name emqx \
  -p 1883:1883 \ 
  -p 8083:8083 \   
  -p 18083:18083 \  
  emqx/emqx:latest
```

## Testing
### Testing Guide
1. Go to Root Project
2. Restart all terminal
3. Open terminal 1
```bash
python -m bridge.mqtt_redis clear
```
3. Open terminal 2, use genetic for GA optimizarion, or pso for PSO optimization
```bash
python -m run_agent.run_agent genetic
```
4. Open terminal 3
5. Run simulation scripts

## List of simulation scripts
### NFRT-01
1. script without scheduling
```bash
python -m scripts.integration 0 n
```
2. script with scheduling
```bash
python -m scripts.integration 0 y
```

### NFRT-02
1. script weight 1
```bash
python -m scripts.integration 1 y
```
2. script weight 2
```bash
python -m scripts.integration 2 y
```
3. script weight 3
```bash
python -m scripts.integration 3 y
```

### NFRT-03
1. script length 1
```bash
python -m scripts.integration 2 y
```
2. script length 2
```bash
python -m scripts.integration 4 y
```
3. script length 3
```bash
python -m scripts.integration 5 y
```