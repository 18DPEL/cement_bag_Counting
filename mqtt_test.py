import paho.mqtt.client as mqtt
import json
import time

# MQTT broker details
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "cement/wagon/target/true"

# Callback when connected to MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

# Callback when message is published
def on_publish(client, userdata, mid):
    print(f"Message published with MID: {mid}")

# Create MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_publish = on_publish

try:
    # Connect to broker
    print("Connecting to broker...")
    client.connect(BROKER, PORT, 60)
    
    # Use loop() instead of loop_start() for synchronous operation
    client.loop(timeout=1.0)  # Process network events
    
    # Create sample data
    data = {
        "cement_type": "OPC-53",
        "wagon_id": "WGN-001",
        "target_weight": 50.0,
        "status": "true",
        "timestamp": time.time(),
        "destination": "Plant A",
        "batch_number": "BATCH-2024-001"
    }
    
    # Convert to JSON and publish
    message = json.dumps(data)
    
    print("Publishing message...")
    result = client.publish(TOPIC, message)
    
    # Process network events to ensure publish
    client.loop(timeout=1.0)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("Message published successfully")
    else:
        print(f"Failed to publish message: {result.rc}")
    
    # Disconnect
    client.disconnect()
    print("Disconnected from MQTT Broker")
    
except Exception as e:
    print(f"Error: {e}")