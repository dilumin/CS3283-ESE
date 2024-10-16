
# Safety Inspection Drone for Construction (Embedded Software Project)

This project develops embedded software for a drone designed to inspect cracks and safety hazards on high-up buildings, bridges, and other structures, minimizing human risk.

**Features:**

* **Crack Detection:** Employs image processing techniques to identify potential cracks in captured images.
* **Safety Hazard Recognition:** Integrates sensors (e.g. thermal cameras) for detecting loose objects, structural weaknesses, and other safety concerns.
* **Data Acquisition and Reporting:** Captures high-resolution images and sensor data for post-inspection analysis and reporting.

**Hardware :**

* Drone platform (quadcopter)
* Raspberry pi Microcontroller 
* Camera (high-resolution for crack detection)
* Additional sensors ( ToF Sensor )
* Communication module (e.g., Wi-Fi ) for data transmission
* Power management system

**Software:**

* Crack identification (Yolo V8 nano)
* Image processing library (OpenCV) for crack detection
* Sensor data processing libraries (specific to sensor types)
* Communication protocol for data transmission 

**Getting Started:**

1. **Hardware Setup:** Assemble and configure the drone platform, sensors, communication module, and power management system.
2. **Software Installation:** Install the embedded operating system, and necessary libraries on your development environment.
3. **Code Development:** 
   * Image capture and processing for crack detection
   * Sensor data acquisition and interpretation
   * Data transmission protocol
  
### Setup

1. **Clone the repository:**
   
    ```bash
    git clone https://github.com/dilumin/CS3283-ESE.git
    cd CS3283-ESE
    ```

**Testing and Deployment:**

1. **Ground Testing:** Thoroughly test the drone's functionalities in a controlled environment before flying outdoors.
2. **Outdoor Testing:** Conduct initial outdoor flights with a safety observer present, gradually increasing complexity.
3. **Data Analysis:** Develop a system for analyzing captured data to identify and assess cracks and safety hazards.

