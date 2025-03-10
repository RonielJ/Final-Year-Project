from fastapi import FastAPI, WebSocket
import subprocess
import asyncio
import traci
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define available environments and corresponding config files
ENVIRONMENTS = {
    "env1": r"/mnt/c/Users/ronie/Downloads/FinalYearProject/TransSimHub/examples/sumo_env/single_junction/env/single_junction.sumocfg",
    "env2": r"/mnt/c/Users/ronie/Downloads/FinalYearProject/TransSimHub/examples/sumo_env/three_junctions/env/3junctions.sumocfg",
    "env3": r"/mnt/c/Users/ronie/Downloads/FinalYearProject/TransSimHub/examples/sumo_env/pedestrian_cross/env/pedestrian_cross.sumocfg",
}

sumo_process = None  # Store the running SUMO process


@app.get("/start_simulation/{env_name}")
async def start_simulation(env_name: str):
    global sumo_process

    if env_name not in ENVIRONMENTS:
        return {"error": "Invalid environment selected"}

    cfg_file = ENVIRONMENTS[env_name]

    # Start SUMO-GUI
    sumo_process = subprocess.Popen(["sumo-gui", "-c", cfg_file])
    return {"status": f"Simulation started for {env_name}"}

    # Wait to ensure SUMO starts before connecting
    try:
        await asyncio.sleep(3)
        traci.start(["sumo-gui", "-c", cfg_file])
        return {"status": f"Simulation started for {env_name}"}
    except Exception as e:
        return {"error": "Failed to start SUMO", "details": str(e)}


    


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            traci.simulationStep()

            # Example: Get number of vehicles on a certain edge
            vehicle_count = traci.edge.getLastStepVehicleNumber("edge_id")

            # Send real-time updates to the frontend
            await websocket.send_json({"vehicle_count": vehicle_count})
            await asyncio.sleep(1)  # Simulate real-time updates

        except Exception as e:
            print(f"Error: {e}")
            break

    traci.close()
