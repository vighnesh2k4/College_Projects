import requests

ESP_IP = "192.168.137.210" 
ESP_PORT = 80              

def send_command(command_path):
    """
    Sends an HTTP GET command to the ESP8266.
    Args:
        command_path (str): The path for the command (e.g., "fan/on").
    """
    url = f"http://{ESP_IP}:{ESP_PORT}/{command_path}"
    try:
        response = requests.get(url, timeout=5) 
        response.raise_for_status() 
        print(f"Command '{command_path}' sent successfully. ESP response: {response.text}")
        return True
    except requests.exceptions.Timeout:
        print(f"Error: Request to ESP8266 timed out for command '{command_path}'.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error sending command '{command_path}' to ESP8266: {e}")
        print("Please ensure the ESP8266 is connected to the same network and its IP address is correct.")
        return False

def direct_command_mode():
    """
    Allows the user to send direct commands to the ESP8266.
    """
    print("\n--- Direct Command Mode ---")
    print("Available commands:")
    print("  - fan_on: Turn fan on")
    print("  - fan_off: Turn fan off")
    print("  - light_on: Turn light on")
    print("  - light_off: Turn light off")
    print("  - exit: Exit direct command mode")

    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == "fan_on":
            send_command("fan/on")
        elif cmd == "fan_off":
            send_command("fan/off")
        elif cmd == "light_on":
            send_command("light/on")
        elif cmd == "light_off":
            send_command("light/off")
        elif cmd == "exit":
            print("Exiting direct command mode.")
            break
        else:
            print("Invalid command. Please try again.")