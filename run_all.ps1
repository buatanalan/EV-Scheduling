Write-Host "Checking Redis container status..."
docker start 28300cb1dc2ad81550bb342589c69143634453bfc6a52387129f2a86817d144d | Out-Null

Write-Host "Checking EQMX broker container status..."
docker start 75f16c2018ad67c42a1329539ea5e92b6d496e18777e4f2b49526942785f19d7 | Out-Null

Write-Host "Starting Python programs..."
Start-Process python -ArgumentList "run_agent/run_agent.py"
Start-Process python -ArgumentList "bridge/mqtt_redis.py"
Start-Process python -ArgumentList "scripts/integration.py 0"
