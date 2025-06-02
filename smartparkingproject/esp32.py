from machine import Pin, PWM, SoftI2C
from time import sleep
import network
import socket
import json
try:
    import sh1106
except ImportError:
    print("sh1106 library not found. Install it or comment out display code.")
    sh1106 = None

# ===== CONFIGURATION =====
SSID = "No_internet"
PASSWORD = "Shiva@234"

# GPIO Pins
SERVO_PIN = 4
BUZZER_PIN = 5
I2C_SCL = 16
I2C_SDA = 17

# IR Sensors for 6 slots
IR_PINS = [14, 2, 18, 19, 27, 26]

# ===== HARDWARE INITIALIZATION =====
servo = PWM(Pin(SERVO_PIN), freq=50)
servo.duty(26)  # Initial position (gate closed)

buzzer = Pin(BUZZER_PIN, Pin.OUT)
buzzer.off()

# IR Sensor Pins
ir_sensors = [Pin(pin, Pin.IN) for pin in IR_PINS]

# OLED Display - SH1106
has_display = False
oled = None

if sh1106:
    try:
        i2c = SoftI2C(scl=Pin(I2C_SCL), sda=Pin(I2C_SDA))
        oled = sh1106.SH1106_I2C(128, 64, i2c)
        has_display = True
        print("OLED display initialized")
    except Exception as e:
        has_display = False
        print("OLED display init failed:", e)

# ===== FUNCTIONS =====
def update_display(message=""):
    if not has_display or not oled:
        return
    try:
        oled.fill(0)
        oled.text("Gate Control", 0, 0)
        
        # Display up to 5 slots (limited by screen space)
        for i in range(min(5, len(ir_sensors))):
            slot_status = "Free" if ir_sensors[i].value() else "Occupied"
            oled.text(f"S{i+1}: {slot_status}", 0, 10 + (i * 9))
        
        # Show slot 6 on same line as slot 5 if space allows
        if len(ir_sensors) > 5:
            slot_status = "Free" if ir_sensors[5].value() else "Occupied"
            oled.text(f"S6:{slot_status}", 70, 46)
        
        if message:
            oled.text(message[:16], 0, 56)
        oled.show()
    except Exception as e:
        print("Display update error:", e)

def connect_wifi():
    update_display("Connecting WiFi")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    max_wait = 20
    while max_wait > 0:
        if wlan.isconnected():
            break
        sleep(1)
        max_wait -= 1
        print(f"Connecting... {max_wait}")
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print("Connected! IP:", ip)
        update_display(f"IP: {ip}")
        sleep(2)
        return ip
    else:
        print("WiFi connection failed")
        update_display("WiFi failed!")
        return None

def set_servo_angle(angle):
    # Clamp angle between 0 and 180
    angle = max(0, min(180, angle))
    duty = int((angle / 180) * 102 + 26)
    servo.duty(duty)

def open_gate():
    print("Opening gate")
    update_display("Opening gate")
    
    # Buzzer beep
    buzzer.on()
    sleep(0.3)
    buzzer.off()
    
    # Open gate (90 degrees)
    set_servo_angle(90)
    sleep(3)  # Keep gate open for 3 seconds
    
    # Close gate
    update_display("Closing gate")
    set_servo_angle(0)
    sleep(1)  # Wait for gate to close
    update_display("Gate Ready")

def get_slots():
    slots = []
    occupied_slots = []
    
    for i, ir in enumerate(ir_sensors):
        is_occupied = not ir.value()  # Assuming LOW = occupied
        slots.append(is_occupied)
        if is_occupied:
            occupied_slots.append(i + 1)
    
    return {
        "slots": slots,
        "occupied_slots": occupied_slots,
        "total_slots": len(ir_sensors),
        "occupied_count": len(occupied_slots)
    }

def create_html_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32 Gate Control Dashboard</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <!-- Animate.css -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        body {
            background: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        .container {
            margin-top: 20px;
        }
        .card {
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .slot-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }
        button {
            transition: transform 0.1s ease;
        }
        button:active {
            transform: scale(0.95);
        }
        .navbar {
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">🏠 Gate Control Dashboard</a>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container">
        <div class="row">
            <!-- Gate Control Section -->
            <div class="col-md-6">
                <h2>Gate Control</h2>
                <button class="btn btn-primary btn-lg mb-3" onclick="operateGate()">🚪 OPEN GATE</button>
                <div id="status" class="alert alert-info" role="alert">Gate Ready</div>
            </div>
            <!-- Parking Slots Section -->
            <div class="col-md-6">
                <h2>Parking Slots</h2>
                <div class="row">
                    <div class="col-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Slot 1</h5>
                                <p class="card-text" id="slot1"><i class="fas fa-parking slot-icon"></i> Loading...</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Slot 2</h5>
                                <p class="card-text" id="slot2"><i class="fas fa-parking slot-icon"></i> Loading...</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Slot 3</h5>
                                <p class="card-text" id="slot3"><i class="fas fa-parking slot-icon"></i> Loading...</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Slot 4</h5>
                                <p class="card-text" id="slot4"><i class="fas fa-parking slot-icon"></i> Loading...</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Slot 5</h5>
                                <p class="card-text" id="slot5"><i class="fas fa-parking slot-icon"></i> Loading...</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Slot 6</h5>
                                <p class="card-text" id="slot6"><i class="fas fa-parking slot-icon"></i> Loading...</p>
                            </div>
                        </div>
                    </div>
                </div>
                <p id="occupiedCount" class="mt-3">Occupied: Loading...</p>
                <button class="btn btn-success btn-sm" onclick="refreshSlots()">🔄 Refresh Slots</button>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function operateGate() {
            const status = document.getElementById('status');
            status.innerText = 'Opening gate...';
            status.className = 'alert alert-warning';
            
            fetch('/open_gate')
                .then(response => {
                    if (response.ok) {
                        status.innerText = 'Gate operated successfully';
                        status.className = 'alert alert-success animate_animated animate_fadeIn';
                        setTimeout(() => {
                            status.innerText = 'Gate Ready';
                            status.className = 'alert alert-info';
                        }, 4000);
                    } else {
                        throw new Error('Server error');
                    }
                })
                .catch(error => {
                    status.innerText = 'Error operating gate';
                    status.className = 'alert alert-danger';
                    console.error('Error:', error);
                });
        }
        
        function refreshSlots() {
            fetch('/get_slots')
                .then(response => response.json())
                .then(data => {
                    for (let i = 0; i < data.total_slots; i++) {
                        const slotElement = document.getElementById(slot${i + 1});
                        const status = data.slots[i] ? 'Occupied' : 'Free';
                        const icon = data.slots[i] ? 'fa-car' : 'fa-parking';
                        const color = data.slots[i] ? 'text-danger' : 'text-success';
                        slotElement.innerHTML = <i class="fas ${icon} slot-icon"></i> ${status};
                        slotElement.className = card-text ${color} animate__animated animate__fadeIn;
                    }
                    document.getElementById('occupiedCount').innerText = Occupied: ${data.occupied_count}/${data.total_slots};
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('occupiedCount').innerText = 'Error loading slot data';
                });
        }
        
        // Auto-refresh every 5 seconds
        setInterval(refreshSlots, 5000);
        
        // Initial load
        refreshSlots();
    </script>
</body>
</html>"""
def handle_client(cl):
    try:
        request = cl.recv(1024).decode('utf-8')
        print("Request:", request.split('\r\n')[0])  # Print first line only
        
        if "GET /open_gate" in request:
            open_gate()
            response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\n\r\nGate operated"
            cl.send(response.encode())
            
        elif "GET /get_slots" in request:
            slots_data = get_slots()
            json_response = json.dumps(slots_data)
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{json_response}"
            cl.send(response.encode())
            
        else:  # Serve main page
            html = create_html_page()
            response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{html}"
            cl.send(response.encode())
            
    except Exception as e:
        print("Client handling error:", e)
    finally:
        try:
            cl.close()
        except:
            pass

def main():
    print("Starting ESP32 Gate Control System...")
    update_display("System Start")
    
    # Initialize servo to closed position
    set_servo_angle(0)
    sleep(1)
    update_display("Gate Ready")
    
    # Connect to WiFi
    ip = connect_wifi()
    if not ip:
        print("Cannot continue without WiFi")
        update_display("No WiFi!")
        return
    
    # Setup web server
    try:
        addr = socket.getaddrinfo(ip, 80)[0][-1]
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(addr)
        s.listen(5)
        s.settimeout(1.0)  # 1 second timeout
        print(f"Web server running at http://{ip}/")
        update_display(f"Server: {ip}")
    except Exception as e:
        print("Server setup error:", e)
        update_display("Server error")
        return
    
    # Main server loop
    print("Server ready. Waiting for connections...")
    while True:
        try:
            # Update display with current slot status
            update_display()
            
            # Handle incoming connections
            try:
                cl, addr = s.accept()
                print(f"Client connected from {addr}")
                handle_client(cl)
            except OSError:
                # Timeout - no connection, continue loop
                pass
                
            sleep(0.1)  # Small delay to prevent overwhelming the system
            
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print("Main loop error:", e)
            sleep(1)
    
    # Cleanup
    try:
        s.close()
    except:
        pass

# Entry point
if _name_ == "_main_":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print("Fatal error:", e)
    finally:
        print("Cleaning up...")
        servo.deinit()
        buzzer.off()
        if has_display and oled:
            try:
                update_display("System Stopped")
                sleep(1)
            except:
                pass
        print("System stopped.")