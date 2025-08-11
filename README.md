# frigatenvr-reporter-addon
A standalone web dashboard and reporting tool for Frigate NVR.

<img width="1641" height="892" alt="image" src="https://github.com/user-attachments/assets/f04b854c-e294-4d20-bf40-04c41cddba7c" />

Frigate Reporter Addon:

A standalone, self-contained script that deploys a web-based dashboard and reporting tool for the Frigate NVR. This tool provides at-a-glance analytics, a visual site map, semantic event searching, LPR event logging, and PDF/CSV report exporting.

The entire application, including the Python backend and web UI, is created and managed by a single bash script.

Features

•	Analytics Dashboard: Visualize detection statistics, including total events, hourly trends, most active cameras, and most frequent objects.
•	Site Map: Upload a floor plan or map and place camera icons to visualize your layout and view live camera status.
•	Explore & Search: Use natural language (semantic search) to find specific events (e.g., "a red car driving down the street").
•	LPR (License Plate Recognizer): View a dedicated log of all captured license plate events.
•	PDF & CSV Export: Generate detailed PDF executive summaries or export raw event data to CSV for further analysis.
•	Easy Installation: Single-script deployment with minimal dependencies.


Prerequisites

This script is designed for Debian-based systems (like Ubuntu, Debian) and requires the following to be installed:
•	Docker: To run the addon container.
•	sqlite3: To perform initial checks on your Frigate database.
The addon automatically uses the "frigate" docker container if using the easy installation script. No additional configuration is needed.

The script will attempt to automatically install these dependencies using apt-get if they are not found.


Installation

1.	Download the frigate_reporter.sh script to your Frigate server.
2.	Make the script executable:
3.	chmod +x frigate_reporter.sh
4.	Run the installer with sudo. It needs root permissions to interact with Docker and install dependencies.
5.	sudo ./frigate_reporter.sh install


Note on SSL Certificate: The script generates a self-signed SSL certificate for HTTPS. When you first access the web UI, your browser will show a security warning. This is expected. Please proceed past the warning to access the application.


Usage

Once installed, you can access the web UI at https://<YOUR_SERVER_IP>:5008.

You can also manage the addon container from your terminal:

•	Start the addon:
•	sudo ./frigate_reporter.sh start

•	Stop the addon:
•	sudo ./frigate_reporter.sh stop

•	Delete the addon (removes container, image, and all files):
•	sudo ./frigate_reporter.sh delete

•	Check your system for dependencies and database access:
•	sudo ./frigate_reporter.sh check


**Additional screenshots:**


Semantic search section:

<img width="1633" height="887" alt="image" src="https://github.com/user-attachments/assets/22eb11db-83fd-4027-b8f8-20f8fd3dbb5b" />

LPR Section:

<img width="1625" height="886" alt="image" src="https://github.com/user-attachments/assets/a5175ad6-c1b3-4de9-b825-435fa5feb6ca" />

Site Map section:

<img width="1622" height="902" alt="image" src="https://github.com/user-attachments/assets/419f2aa2-3790-422f-9f0a-e81bb9bf945a" />

Additional Dashboard:

<img width="1599" height="904" alt="image" src="https://github.com/user-attachments/assets/67317938-2124-4ebd-9a4b-6bb698336321" />

<img width="1635" height="753" alt="image" src="https://github.com/user-attachments/assets/15195295-de55-4f7d-90a8-b4d424dff2f7" />

Executive PDF Reports:

<img width="1654" height="890" alt="image" src="https://github.com/user-attachments/assets/aa1f368a-5614-45f0-93d9-84e5b1029541" />

Example Report:

[frigate_executive_summary.pdf](https://github.com/user-attachments/files/21723661/frigate_executive_summary.pdf)




Disclaimer & About

This is a community-developed project and is not officially supported by the Frigate team. It was created as a proof-of-concept to explore what was possible.
This project is provided "as is" and without any warranty. The author is not responsible for any damage or loss caused by its use. You are using this software at your own risk.
This script was developed in collaboration with a large language model (Google's Gemini). The entire process, from initial code generation to debugging and refinement, was guided and validated by Google Gemini, and end users.

**This is a community-developed project and is not officially supported by the Frigate team.** 

