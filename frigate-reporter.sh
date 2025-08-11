#!/bin/bash

# ==============================================================================
# Frigate Reporter Addon Script
#
# Author: Gemini
# Version: 5.0.1 (Final Thumbnail Fix)
# Fixes:   - Corrected search result thumbnail URL generation to use the event ID,
#            resolving broken images.
# ==============================================================================

# Color definitions for professional output
COLOR_RESET=$'\e[0m'
COLOR_BOLD=$'\e[1m'
COLOR_GREEN=$'\e[32m'
COLOR_YELLOW=$'\e[33m'
COLOR_RED=$'\e[31m'
COLOR_HEADER=$'\e[1;33m'
COLOR_INFO=$'\e[32m'
COLOR_SUCCESS=$'\e[1;32m'
COLOR_WARN=$'\e[33m'
COLOR_PROMPT=$'\e[33m'
COLOR_ERROR=$'\e[31m'

# Function for section headers
section_header() {
    echo -e "\n${COLOR_HEADER}================================================================================${COLOR_RESET}" >&2
    echo -e "${COLOR_HEADER} $1 ${COLOR_RESET}" >&2
    echo -e "${COLOR_HEADER}================================================================================${COLOR_RESET}\n" >&2
}

success_msg() { echo -e "${COLOR_SUCCESS}[SUCCESS] $1${COLOR_RESET}" >&2; }
info_msg() { echo -e "${COLOR_INFO}[INFO] $1${COLOR_RESET}" >&2; }
warn_msg() { echo -e "${COLOR_WARN}[WARNING] $1${COLOR_RESET}" >&2; }
error_msg() { echo -e "${COLOR_ERROR}[ERROR] $1${COLOR_RESET}" >&2; }

# --- Script Configuration ---
ADDON_DIR="./frigate_reporter_addon"
DOCKER_IMAGE_NAME="frigate-report-addon"
DOCKER_CONTAINER_NAME="frigate-reporter"
FRIGATE_CONTAINER_NAME="frigate"
ADDON_PORT="5008" # Port for host networking

# --- Helper Functions ---
check_dependencies() {
    section_header "Checking & Installing Dependencies"
    info_msg "This script will attempt to install any missing dependencies using 'apt-get'."
    
    declare -A deps=(
        ["docker"]="docker.io"
        ["sqlite3"]="sqlite3"
        ["wget"]="wget"
        ["openssl"]="openssl"
    )
    local missing_deps=0
    
    if ! apt-get -v &>/dev/null; then
        error_msg "'apt-get' not found. Cannot automatically install dependencies. Please ensure they are installed manually."
        return 1
    fi
    
    info_msg "Updating package lists..."
    if ! sudo apt-get update -y; then
        error_msg "Failed to update package lists. Please check your internet connection and repository configuration."
        exit 1
    fi

    for cmd in "${!deps[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            local pkg_name=${deps[$cmd]}
            warn_msg "'$cmd' is not installed. Attempting to install '$pkg_name'..."
            if ! sudo apt-get install -y "$pkg_name"; then
                 error_msg "Failed to install '$pkg_name'. Please try installing it manually and re-run the script."
                 missing_deps=1
            else
                 success_msg "'$pkg_name' has been installed successfully."
            fi
        else
            success_msg "'$cmd' is installed."
        fi
    done
    
    info_msg "Checking for PDF rendering libraries (pango)..."
    if ! dpkg -s libpango-1.0-0 &>/dev/null; then
        warn_msg "PDF library 'pango' not found. Attempting to install..."
        if ! sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0; then
            error_msg "Failed to install PDF libraries. PDF export may fail."
            missing_deps=1
        else
            success_msg "PDF libraries installed."
        fi
    else
        success_msg "PDF libraries are installed."
    fi


    if [ "$missing_deps" -eq 1 ]; then 
        error_msg "One or more critical dependencies could not be installed. Exiting."
        exit 1
    fi
}

find_frigate_data_path() {
    section_header "Locating Frigate Data Directory"
    info_msg "Inspecting the '${FRIGATE_CONTAINER_NAME}' container to find data path..."
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${FRIGATE_CONTAINER_NAME}$"; then
        error_msg "Frigate container '${FRIGATE_CONTAINER_NAME}' not found. Is it running?"
        exit 1
    fi

    FRIGATE_DATA_HOST_PATH=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/config"}}{{.Source}}{{end}}{{end}}' ${FRIGATE_CONTAINER_NAME})

    if [ -z "$FRIGATE_DATA_HOST_PATH" ]; then
        error_msg "Could not find Frigate's '/config' volume mount."
        info_msg "Please ensure your Frigate container has a volume mapped to '/config'."
        exit 1
    fi
    success_msg "Found Frigate data host directory: ${FRIGATE_DATA_HOST_PATH}"

    FRIGATE_DB_PATH="${FRIGATE_DATA_HOST_PATH}/frigate.db"

    if [ ! -f "${FRIGATE_DB_PATH}" ]; then
        error_msg "Could not find frigate.db in ${FRIGATE_DATA_HOST_PATH}"
        exit 1
    fi
    success_msg "Found database file: ${FRIGATE_DB_PATH}"
}

check_system() {
    section_header "Running System Check"
    check_dependencies
    find_frigate_data_path

    info_msg "Checking for 'event' table in ${FRIGATE_DB_PATH}..."
    
    local table_check_output
    table_check_output=$(timeout 5 sqlite3 "${FRIGATE_DB_PATH}" ".tables" 2>/dev/null)
    
    if [[ -z "$table_check_output" ]]; then
        error_msg "Could not read from the database. It might be locked or permissions are incorrect."
        info_msg "Try stopping your main Frigate container temporarily and re-running this check."
        exit 1
    fi

    if echo "$table_check_output" | grep -wq "event"; then
        success_msg "OK: Found 'event' table in the database."
    else
        error_msg "CRITICAL: The 'event' table was NOT found in your frigate.db file!"
        warn_msg "This indicates a problem with your main Frigate installation's database, not this addon."
        info_msg "Your database contains the following tables: ${table_check_output}"
    fi
}


# --- Main Application Logic ---
create_addon_files() {
    section_header "Creating Reporter Application Files"
    info_msg "Creating directory and files at ${ADDON_DIR}"
    mkdir -p "${ADDON_DIR}/templates"
    mkdir -p "${ADDON_DIR}/certs"
    mkdir -p "${ADDON_DIR}/storage" # For map image and layout

    if [ ! -f "${ADDON_DIR}/certs/key.pem" ] || [ ! -f "${ADDON_DIR}/certs/cert.pem" ]; then
        info_msg "Generating self-signed certificate for HTTPS..."
        openssl req -x509 -newkey rsa:4096 -nodes \
            -keyout "${ADDON_DIR}/certs/key.pem" \
            -out "${ADDON_DIR}/certs/cert.pem" \
            -sha256 -days 3650 \
            -subj "/C=US/ST=State/L=City/O=Frigate/OU=Reporter/CN=localhost"
        success_msg "Certificate generated."
    else
        info_msg "Existing certificate found. Skipping generation."
    fi

    cat <<'EOF' > "${ADDON_DIR}/Dockerfile"
FROM python:3.9-slim
WORKDIR /app
# Disable io_uring for SQLite to prevent potential DB lock issues on some systems
ENV SQLITE_DISABLE_IO_URING=1
# Install PDF rendering libraries first
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Install Python packages
RUN pip install --no-cache-dir Flask requests weasyprint PyYAML Pillow
COPY . .
# Set the default command to run when the container starts
CMD ["python3", "app.py"]
EOF
    success_msg "Created Dockerfile."

    cat <<'EOF' > "${ADDON_DIR}/app.py"
import os
import sqlite3
import time
import logging
import json
import csv
import base64
from io import BytesIO, StringIO
from datetime import datetime
from collections import Counter, defaultdict
from itertools import groupby
import requests
import yaml
from flask import Flask, jsonify, render_template, request, Response, stream_with_context, send_from_directory
from weasyprint import HTML, CSS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG_DIR = '/config'
STORAGE_DIR = '/app/storage'
DB_PATH = os.path.join(CONFIG_DIR, 'frigate.db')
CONFIG_PATH_YML = os.path.join(CONFIG_DIR, 'config.yml')
CERT_FILE = '/app/certs/cert.pem'
KEY_FILE = '/app/certs/key.pem'
FRIGATE_BASE_URL = 'http' + ('s' if os.environ.get('FRIGATE_HTTPS') else '') + '://' + os.environ.get('FRIGATE_HOST_IP', '127.0.0.1') + ':5000'
MAP_LAYOUT_FILE = os.path.join(STORAGE_DIR, 'map_layout.json')
MAP_IMAGE_FILE = 'sitemap.jpg'
ADDON_PORT = 5008 # The port the app will run on inside the container

app.config['UPLOAD_FOLDER'] = STORAGE_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit for map image

def db_connect():
    try:
        # Connect in read-only mode to prevent db locks
        conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        app.logger.error(f"Database connection error: {e}")
        return None

def get_query_filters(request_args, prefix=""):
    start_key = f"{prefix}start"
    end_key = f"{prefix}end"
    
    start_time_epoch = float(request_args.get(start_key, time.time() - 86400 * 30))
    end_time_epoch = float(request_args.get(end_key, time.time()))
    
    selected_cameras_str = request_args.get('cameras')
    selected_cameras = selected_cameras_str.split(',') if selected_cameras_str else []
    
    params = [start_time_epoch, end_time_epoch]
    camera_clause = ""
    if selected_cameras:
        placeholders = ','.join('?' for _ in selected_cameras)
        camera_clause = f"AND camera IN ({placeholders})"
        params.extend(selected_cameras)
        
    return f"start_time >= ? AND start_time <= ? {camera_clause}", tuple(params), (start_time_epoch, end_time_epoch)

def format_duration(seconds):
    if seconds is None: return "N/A"
    s = int(seconds)
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'

def format_timestamp(ts):
    if ts is None: return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config_status')
def get_config_status():
    status = {'semantic_search': False, 'lpr': False}
    try:
        if os.path.exists(CONFIG_PATH_YML):
            with open(CONFIG_PATH_YML, 'r') as f:
                config = yaml.safe_load(f)
                if config.get('semantic_search', {}).get('enabled', False):
                    status['semantic_search'] = True
                if config.get('lpr', {}).get('enabled', False):
                    status['lpr'] = True
    except Exception as e:
        app.logger.error(f"Error reading Frigate config file: {e}")
    return jsonify(status)

@app.route('/api/cameras')
def get_cameras():
    conn = db_connect()
    if not conn: return jsonify({"error": "Could not connect to the Frigate database."}), 500
    try:
        cursor = conn.cursor()
        query = "SELECT DISTINCT camera FROM event ORDER BY camera;"
        cursor.execute(query)
        cameras = [row['camera'] for row in cursor.fetchall()]
        return jsonify(cameras)
    except sqlite3.Error as e:
        app.logger.error(f"DB Error getting cameras: {e}")
        return jsonify({"error": f"DB Error getting cameras: {e}"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/camera_status/<camera_name>')
def get_camera_status(camera_name):
    conn = db_connect()
    if not conn: return jsonify({"error": "DB connection error"}), 500
    try:
        cursor = conn.cursor()
        now = time.time()
        # Get total detections in the last 24 hours for this camera
        query = "SELECT COUNT(id) as count FROM event WHERE camera = ? AND start_time >= ?;"
        cursor.execute(query, (camera_name, now - 86400))
        last_24h_detections = cursor.fetchone()['count']

        # Get the last detection time
        query = "SELECT start_time FROM event WHERE camera = ? ORDER BY start_time DESC LIMIT 1;"
        cursor.execute(query, (camera_name,))
        last_detection_row = cursor.fetchone()
        last_detection = format_timestamp(last_detection_row['start_time']) if last_detection_row else "Never"

        return jsonify({
            "name": camera_name,
            "detections_24h": last_24h_detections,
            "last_detection_time": last_detection,
            "snapshot_url": f"/api/frigate_proxy/api/{camera_name}/latest.jpg?t={now}"
        })
    except sqlite3.Error as e:
        app.logger.error(f"DB Error getting camera status for {camera_name}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: conn.close()

def get_stats_data(request_args):
    # Main filters for everything except hourly chart override
    where_clause, params, (start_time, end_time) = get_query_filters(request_args)
    
    # Specific filters for hourly chart if provided
    hourly_where_clause, hourly_params, (hourly_start, hourly_end) = get_query_filters(request_args, prefix="hourly_")

    conn = db_connect()
    if not conn: return None
    
    try:
        cursor = conn.cursor()
        
        # --- STAT PANELS (use main filters) ---
        total_detections_q = f"SELECT COUNT(id) as count FROM event WHERE {where_clause};"
        cursor.execute(total_detections_q, params)
        total_detections = cursor.fetchone()['count']

        duration = end_time - start_time
        prev_start_time = start_time - duration
        prev_params = list(params); prev_params[0] = prev_start_time; prev_params[1] = start_time
        prev_total_q = f"SELECT COUNT(id) as count FROM event WHERE {where_clause};"
        cursor.execute(prev_total_q, tuple(prev_params))
        prev_total = cursor.fetchone()['count']
        anomaly_percent = ((total_detections - prev_total) / prev_total) * 100 if prev_total > 0 else (100 if total_detections > 0 else 0)

        most_active_cam_q = f"SELECT camera, COUNT(id) as count FROM event WHERE {where_clause} GROUP BY camera ORDER BY count DESC LIMIT 1;"
        cursor.execute(most_active_cam_q, params)
        most_active_cam = dict(cursor.fetchone()) if total_detections > 0 else {}

        most_freq_obj_q = f"SELECT label, COUNT(id) as count FROM event WHERE {where_clause} GROUP BY label ORDER BY count DESC LIMIT 1;"
        cursor.execute(most_freq_obj_q, params)
        most_freq_obj = dict(cursor.fetchone()) if total_detections > 0 else {}
        
        busiest_hour_q = f"SELECT strftime('%H', datetime(start_time, 'unixepoch')) as hour, COUNT(id) as count FROM event WHERE {where_clause} GROUP BY hour ORDER BY count DESC LIMIT 1;"
        cursor.execute(busiest_hour_q, params)
        busiest_hour = dict(cursor.fetchone()) if total_detections > 0 else {}

        # --- HOURLY CHARTS (use hourly filters) ---
        hourly_query = f"SELECT strftime('%H', datetime(start_time, 'unixepoch')) as hour, label, COUNT(id) as count FROM event WHERE {hourly_where_clause} GROUP BY hour, label;"
        cursor.execute(hourly_query, hourly_params)
        hourly_trends = defaultdict(dict)
        for row in cursor.fetchall():
            hourly_trends[row['hour']][row['label']] = row['count']

        hourly_duration = hourly_end - hourly_start
        prev_hourly_start = hourly_start - hourly_duration
        prev_hourly_params = list(hourly_params); prev_hourly_params[0] = prev_hourly_start; prev_hourly_params[1] = hourly_start
        prev_hourly_query = f"SELECT strftime('%H', datetime(start_time, 'unixepoch')) as hour, COUNT(id) as count FROM event WHERE {hourly_where_clause} GROUP BY hour;"
        cursor.execute(prev_hourly_query, tuple(prev_hourly_params))
        prev_hourly_trends = {row['hour']: row['count'] for row in cursor.fetchall()}
        
        # --- OTHER TABLES (use main filters) ---
        stats_by_cam_zone_q = f"SELECT camera, json_extract(data, '$.zones') as zones, label FROM event WHERE {where_clause};"
        cursor.execute(stats_by_cam_zone_q, params)
        stats_by_camera_zone = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for row in cursor.fetchall():
            try:
                zones = json.loads(row['zones'])
                for zone in zones:
                    stats_by_camera_zone[row['camera']][zone][row['label']] += 1
            except (json.JSONDecodeError, TypeError):
                continue

        dwell_time_q = f"SELECT id, camera, label, start_time, (end_time - start_time) as duration FROM event WHERE end_time IS NOT NULL AND {where_clause} ORDER BY duration DESC LIMIT 10;"
        cursor.execute(dwell_time_q, params)
        longest_events = [dict(row) for row in cursor.fetchall()]

        transition_q = f"SELECT id, camera, start_time FROM event WHERE {where_clause} ORDER BY id, start_time;"
        cursor.execute(transition_q, params)
        camera_transitions = Counter()
        get_root_id = lambda x: x['id'].split('-')[0]
        events_by_id = defaultdict(list)
        for row in cursor.fetchall():
            events_by_id[get_root_id(dict(row))].append(dict(row))

        for event_id, event_group in events_by_id.items():
            path = sorted(event_group, key=lambda x: x['start_time'])
            if len(path) > 1:
                for i in range(len(path) - 1):
                    from_cam, to_cam = path[i]['camera'], path[i+1]['camera']
                    if from_cam != to_cam: camera_transitions[(from_cam, to_cam)] += 1
        
        cursor.close()
        return {
            "stat_panels": {"total_detections": total_detections, "anomaly_percent": anomaly_percent, "most_active_camera": most_active_cam, "most_frequent_object": most_freq_obj, "busiest_hour": busiest_hour},
            "hourly_trends": hourly_trends,
            "prev_hourly_trends": prev_hourly_trends,
            "stats_by_camera_zone": stats_by_camera_zone,
            "longest_events": longest_events,
            "camera_transitions": [{"from": t[0], "to": t[1], "count": c} for t, c in camera_transitions.most_common(10)],
            "date_range": {"start": format_timestamp(start_time), "end": format_timestamp(end_time)},
            "hourly_date_range": {"start": format_timestamp(hourly_start), "end": format_timestamp(hourly_end)}
        }
    except sqlite3.Error as e:
        app.logger.error(f"Database query error in get_stats_data: {e}")
        return {"error": f"Database query error: {e}"}
    finally:
        if conn: conn.close()

@app.route('/api/stats')
def get_stats():
    data = get_stats_data(request.args)
    if data is None or "error" in data:
        return jsonify(data or {"error": "Unknown error occurred while fetching stats"}), 500
    return jsonify(data)

def get_lpr_data(request_args, limit=None):
    where_clause, params, (start_time, end_time) = get_query_filters(request_args)
    conn = db_connect()
    if not conn: return None, None
    
    try:
        cursor = conn.cursor()
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        query = f"""
            SELECT 
                id, camera, start_time, 
                json_extract(data, '$.recognized_license_plate') as plate
            FROM event 
            WHERE 
                label = 'car' AND 
                json_extract(data, '$.recognized_license_plate') IS NOT NULL AND 
                {where_clause} 
            ORDER BY start_time DESC
            {limit_clause};
        """
        cursor.execute(query, params)
        plates = [dict(row) for row in cursor.fetchall()]
        date_range = {"start": format_timestamp(start_time), "end": format_timestamp(end_time)}
        return plates, date_range
    except Exception as e:
        app.logger.error(f"LPR Data Error: {e}")
        return None, None
    finally:
        if conn: conn.close()

@app.route('/api/lpr')
def get_lpr_events():
    plates, _ = get_lpr_data(request.args) # No limit for web UI
    if plates is None:
        return jsonify({"error": "Failed to fetch LPR data"}), 500
    return jsonify(plates)

@app.route('/api/heatmap/<camera_name>')
def get_heatmap(camera_name):
    where_clause, params, _ = get_query_filters(request.args)
    final_params = [params[0], params[1], camera_name]
    final_where = "start_time >= ? AND start_time <= ? AND camera = ?"
    
    conn = db_connect()
    if not conn: return jsonify({"error": "DB connection error"}), 500
    try:
        cursor = conn.cursor()
        query = f"SELECT json_extract(data, '$.box') as box FROM event WHERE {final_where};"
        cursor.execute(query, tuple(final_params))
        points = []
        for row in cursor.fetchall():
            if not row['box']: continue
            try:
                box = json.loads(row['box'])
                if len(box) == 4:
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    points.append({'x': x + w//2, 'y': y + h//2, 'value': 1})
            except (json.JSONDecodeError, IndexError, TypeError):
                app.logger.warning(f"Could not parse box data for heatmap: {row['box']}")
                continue
        return jsonify(points)
    except Exception as e:
        app.logger.error(f"Heatmap Generation Error for camera {camera_name}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: conn.close()

@app.route('/api/search')
def semantic_search():
    query = request.args.get('q')
    if not query: return jsonify({"error": "Query parameter 'q' is required"}), 400
    try:
        # Corrected: Use URL parameters for the search query
        frigate_api_url = f"{FRIGATE_BASE_URL}/api/events/search"
        params = {'query': query}
        app.logger.info(f"Proxying semantic search to: {frigate_api_url} with params {params}")
        response = requests.get(frigate_api_url, params=params, timeout=30, verify=False)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Semantic Search Proxy Error: {e}")
        return jsonify({"error": f"Failed to connect to Frigate for search: {e}"}), 502

@app.route('/api/map/layout', methods=['GET', 'POST'])
def map_layout():
    if request.method == 'POST':
        try:
            with open(MAP_LAYOUT_FILE, 'w') as f:
                json.dump(request.json, f)
            return jsonify({"message": "Layout saved successfully"})
        except Exception as e:
            app.logger.error(f"Error saving map layout: {e}")
            return jsonify({"error": str(e)}), 500
    else: # GET
        if os.path.exists(MAP_LAYOUT_FILE):
            try:
                with open(MAP_LAYOUT_FILE, 'r') as f:
                    # Ensure default structure if file is old
                    data = json.load(f)
                    if 'cameras' not in data:
                        data = {'cameras': data, 'siteName': '', 'siteAddress': ''}
                    return jsonify(data)
            except Exception as e:
                app.logger.error(f"Error loading map layout: {e}")
                return jsonify({'cameras': {}, 'siteName': '', 'siteAddress': ''})
        return jsonify({'cameras': {}, 'siteName': '', 'siteAddress': ''})

@app.route('/api/map/image', methods=['POST'])
def upload_map_image():
    if 'mapImage' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['mapImage']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400
    if file:
        try:
            filename = secure_filename(MAP_IMAGE_FILE)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.logger.info(f"Map image uploaded and saved to {file_path}")
            return jsonify({"message": "Image uploaded successfully", "path": f"/storage/{filename}?t={time.time()}"})
        except Exception as e:
            app.logger.error(f"Failed to save uploaded map image: {e}")
            return jsonify({"error": f"Could not save image: {e}"}), 500

@app.route('/storage/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    try:
        form_data = request.get_json()
        report_data = {}

        # 1. Fetch main stats data
        stats_data = get_stats_data(form_data)
        if not stats_data or "error" in stats_data:
            return "Error fetching data for PDF report", 500
        report_data.update(stats_data)

        # 2. Add chart image and selected sections
        report_data['trends_chart_image'] = form_data.get('trendsChartImage')
        report_data['report_sections'] = form_data.get('sections', {})

        # 3. Fetch LPR data if requested
        if report_data['report_sections'].get('lpr_report'):
            lpr_plates, lpr_range = get_lpr_data(form_data.get('lprFilters', {}), limit=50)
            report_data['lpr_events'] = lpr_plates or []
            report_data['lpr_date_range'] = lpr_range or {}
            for event in report_data['lpr_events']:
                event['start_time_formatted'] = format_timestamp(event.get('start_time'))
                try:
                    thumb_url = f"{FRIGATE_BASE_URL}/api/events/{event['id']}/thumbnail.jpg"
                    response = requests.get(thumb_url, timeout=5, verify=False)
                    if response.status_code == 200:
                        encoded_thumb = base64.b64encode(response.content).decode('utf-8')
                        event['thumbnail_base64'] = f"data:image/jpeg;base64,{encoded_thumb}"
                except Exception as e:
                    app.logger.error(f"Could not fetch thumbnail for event {event['id']}: {e}")

        # 4. Add Site Map data if requested
        if report_data['report_sections'].get('site_map'):
            report_data['site_map_data'] = form_data.get('siteMapData', {})
            # The siteMapImage is now a direct base64 string from the frontend
            report_data['site_map_data']['siteMapImage'] = form_data.get('siteMapImage')

        # 5. Format durations for longest events
        for event in report_data.get('longest_events', []):
            event['duration_formatted'] = format_duration(event.get('duration'))
            event['start_time_formatted'] = format_timestamp(event.get('start_time'))

        # 6. Render HTML and generate PDF
        html_out = render_template('report.html', data=report_data)
        pdf_file = HTML(string=html_out, base_url=request.base_url).write_pdf()
        
        return Response(pdf_file, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=frigate_executive_summary.pdf'})
    except Exception as e:
        app.logger.error(f"PDF Export failed: {e}")
        return f"An internal error occurred during PDF generation: {e}", 500

@app.route('/api/export/csv')
def export_csv():
    where_clause, params, _ = get_query_filters(request.args)
    conn = db_connect()
    if not conn: return "DB connection error", 500

    def generate_csv():
        try:
            cursor = conn.cursor()
            query = f"SELECT camera, label, start_time, end_time, (end_time - start_time) as duration, json_extract(data, '$.zones') as zones FROM event WHERE {where_clause} ORDER BY start_time;"
            cursor.execute(query, params)
            
            output = StringIO()
            writer = csv.writer(output)
            
            writer.writerow(['camera', 'label', 'start_time', 'end_time', 'duration_seconds', 'duration_hh_mm_ss', 'zones'])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

            for row in cursor.fetchall():
                writer.writerow([
                    row['camera'], row['label'],
                    format_timestamp(row['start_time']),
                    format_timestamp(row['end_time']),
                    row['duration'],
                    format_duration(row['duration']),
                    row['zones']
                ])
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)
        finally:
            if conn: conn.close()

    headers = {"Content-disposition": "attachment; filename=frigate_events.csv"}
    return Response(stream_with_context(generate_csv()), mimetype="text/csv", headers=headers)

@app.route('/api/frigate_proxy/<path:path>')
def proxy_frigate(path):
    frigate_url = f"{FRIGATE_BASE_URL}/{path}"
    try:
        req = requests.get(frigate_url, stream=True, timeout=10, params=request.args, verify=False)
        return Response(stream_with_context(req.iter_content(chunk_size=1024)), content_type=req.headers['content-type'])
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to Frigate path '{path}': {e}")
        return "Error connecting to Frigate", 502

if __name__ == '__main__':
    app.logger.info("Starting Frigate Reporter Application...")
    app.logger.info(f"Attempting to connect to Frigate at: {FRIGATE_BASE_URL}")
    if not os.path.exists(DB_PATH):
        logging.critical(f"FATAL: Database file not found at {DB_PATH}. Ensure the volume is mounted correctly. Exiting.")
        exit(1)
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        logging.critical(f"FATAL: SSL Certificate or Key not found in /app/certs. Ensure the volume is mounted correctly. Exiting.")
        exit(1)
        
    app.logger.info(f"Frigate Reporter is ready to serve on port {ADDON_PORT}")
    app.run(host='0.0.0.0', port=ADDON_PORT, ssl_context=(CERT_FILE, KEY_FILE), debug=False)
EOF
    success_msg "Created app.py."

    cat <<'EOF' > "${ADDON_DIR}/templates/report.html"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Frigate Executive Summary</title>
    <style>
        @page {
            size: letter;
            margin: 1in;
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10px;
                color: #666;
            }
        }
        body { font-family: sans-serif; color: #333; }
        h1, h2, h3 { color: #11191f; }
        h1 { font-size: 28px; text-align: center; margin-bottom: 0; }
        h2 { font-size: 20px; border-bottom: 2px solid #38bdf8; padding-bottom: 5px; margin-top: 30px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header .subtitle { font-size: 14px; color: #555; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 20px; margin-top: 20px; }
        .stat-panel { border: 1px solid #ccc; border-radius: 5px; padding: 15px; text-align: center; background-color: #f9f9f9; }
        .stat-panel .value { font-size: 32px; font-weight: bold; color: #38bdf8; }
        .stat-panel .label { font-size: 12px; color: #666; text-transform: uppercase; }
        .chart-container { text-align: center; margin-top: 20px; }
        .chart-container img { max-width: 100%; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 10px; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: middle; }
        th { background-color: #eef; font-size: 11px; }
        .footer {
            position: fixed;
            bottom: -0.8in;
            left: 0;
            right: 0;
            height: 0.5in;
            text-align: left;
            font-size: 10px;
            color: #aaa;
        }
        .confidential { font-style: italic; }
        .page-break { page-break-before: always; }
        .site-map-container {
            position: relative;
            width: 100%;
            border: 1px solid #ccc;
            margin-top: 20px;
        }
        .site-map-container img {
            width: 100%;
            display: block;
        }
        .lpr-thumbnail {
            max-width: 100px;
            max-height: 56px;
            object-fit: cover;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Analytics Report</h1>
        <div class="subtitle">
            For period: {{ data.date_range.start }} to {{ data.date_range.end }}
        </div>
    </div>

    {% if data.report_sections.get('summary_panels', true) %}
    <h2>At-a-Glance Summary</h2>
    <div class="stat-grid">
        <div class="stat-panel">
            <div class="value">{{ data.stat_panels.total_detections }}</div>
            <div class="label">Total Detections</div>
        </div>
        <div class="stat-panel">
            <div class="value" style="color: {{ '#d0342c' if data.stat_panels.anomaly_percent > 10 else '#228b22' }}">{{ '%.1f'|format(data.stat_panels.anomaly_percent) }}%</div>
            <div class="label">vs. Prior Period</div>
        </div>
        <div class="stat-panel">
            <div class="value">{{ data.stat_panels.most_active_camera.camera or 'N/A' }}</div>
            <div class="label">Most Active Camera</div>
        </div>
        <div class="stat-panel">
            <div class="value">{{ data.stat_panels.most_frequent_object.label or 'N/A' }}</div>
            <div class="label">Most Frequent Object</div>
        </div>
    </div>
    {% endif %}

    {% if data.report_sections.get('hourly_trend', true) %}
    <h2>Hourly Activity Trend</h2>
    <div class="subtitle" style="text-align: left; margin-bottom: 10px; margin-top: -10px;">
        For period: {{ data.date_range.start }} to {{ data.date_range.end }}
    </div>
    <div class="chart-container">
        {% if data.trends_chart_image %}
            <img src="{{ data.trends_chart_image }}">
        {% else %}
            <p>Chart data not available.</p>
        {% endif %}
    </div>
    {% endif %}

    {% if data.report_sections.get('dwell_time', true) %}
    <div class="page-break"></div>
    <h2>Anomalous Activity: Longest Dwell Times</h2>
    <table>
        <thead>
            <tr>
                <th>Start Time</th>
                <th>Object</th>
                <th>Camera</th>
                <th>Duration</th>
            </tr>
        </thead>
        <tbody>
            {% for event in data.longest_events %}
            <tr>
                <td>{{ event.start_time_formatted }}</td>
                <td>{{ event.label }}</td>
                <td>{{ event.camera }}</td>
                <td>{{ event.duration_formatted }}</td>
            </tr>
            {% else %}
            <tr><td colspan="4">No events with duration found in this period.</td></tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}

    {% if data.report_sections.get('camera_transitions', true) %}
    <h2>Camera Transition Analysis</h2>
    <table>
        <thead>
            <tr>
                <th>From Camera</th>
                <th>To Camera</th>
                <th>Transition Count</th>
            </tr>
        </thead>
        <tbody>
            {% for t in data.camera_transitions %}
            <tr>
                <td>{{ t.from }}</td>
                <td>{{ t.to }}</td>
                <td>{{ t.count }}</td>
            </tr>
            {% else %}
            <tr><td colspan="3">No camera transition data found.</td></tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}

    {% if data.report_sections.get('site_map', true) and data.site_map_data %}
    <div class="page-break"></div>
    <h2>Site Map</h2>
    <h3>{{ data.site_map_data.siteName or 'N/A' }}</h3>
    <p style="margin-top: -10px; font-size: 12px; color: #555;">{{ data.site_map_data.siteAddress or 'N/A' }}</p>
    <div class="site-map-container">
        {% if data.site_map_data.siteMapImage %}
            <img src="{{ data.site_map_data.siteMapImage }}">
        {% endif %}
    </div>
    {% endif %}

    {% if data.report_sections.get('lpr_report', true) %}
    <div class="page-break"></div>
    <h2>License Plate Recognizer Report</h2>
    <div class="subtitle" style="text-align: left; margin-bottom: 10px;">
        For period: {{ data.lpr_date_range.start }} to {{ data.lpr_date_range.end }}
    </div>
    <table>
        <thead>
            <tr>
                <th>Thumbnail</th>
                <th>Timestamp</th>
                <th>License Plate</th>
                <th>Camera</th>
            </tr>
        </thead>
        <tbody>
            {% for event in data.lpr_events %}
            <tr>
                <td>
                    {% if event.thumbnail_base64 %}
                        <img src="{{ event.thumbnail_base64 }}" class="lpr-thumbnail">
                    {% endif %}
                </td>
                <td>{{ event.start_time_formatted }}</td>
                <td>{{ event.plate }}</td>
                <td>{{ event.camera }}</td>
            </tr>
            {% else %}
            <tr><td colspan="4">No LPR events found in this period.</td></tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    
    <div class="footer">
        <p class="confidential">This document is confidential and intended for internal use only.</p>
    </div>
</body>
</html>
EOF
    success_msg "Created report.html."

    cat <<'EOF' > "${ADDON_DIR}/templates/index.html"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frigate Reporter</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
    <script src="https://cdn.jsdelivr.net/npm/heatmap.js@2.0.5/heatmap.min.js"></script>
    <script src="https://unpkg.com/@panzoom/panzoom@4.5.1/dist/panzoom.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        :root {
            --pico-font-size: 100%; --pico-background-color: #11191f; --pico-color: #dce3e8;
            --pico-h2-color: #f1f5f9; --pico-card-background-color: #1e293b; --pico-card-border-color: #334155;
            --pico-form-element-background-color: #334155; --pico-form-element-border-color: #475569;
            --pico-form-element-focus-border-color: #38bdf8;
        }
        body { padding: 2rem 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .container { max-width: 1600px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; }
        .card { background-color: var(--pico-card-background-color); border: 1px solid var(--pico-card-border-color); padding: 1.5rem; border-radius: var(--pico-border-radius); }
        #status-bar { position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 0.5rem; z-index: 1000; color: white; background-color: #475569;}
        .chart-container { position: relative; height: 300px; width: 100%; margin-top: 1rem; }
        h1, h2, h3 { color: var(--pico-h2-color); border-bottom: 1px solid var(--pico-card-border-color); padding-bottom: 0.5rem; margin-bottom: 1rem; }
        details[role="list"] { width: 100%; }
        details[role="list"] ul { background-color: var(--pico-card-background-color); max-height: 200px; overflow-y: auto; padding: 0.5rem; }
        .stat-panel { text-align: center; }
        .stat-panel .value { font-size: 2.5rem; font-weight: bold; color: #38bdf8; line-height: 1.1; }
        .stat-panel .label { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.5rem; }
        .flatpickr-input { background-color: var(--pico-form-element-background-color); border-color: var(--pico-form-element-border-color); color: var(--pico-color); }
        table { width: 100%; font-size: 0.9rem; }
        .modal-body { padding: 1rem 0; }
        .modal-body label { display: block; margin-bottom: 0.5rem; }
        .tabs [role="tab"] { background-color: var(--pico-form-element-background-color); position: relative; }
        .tabs [role="tab"][aria-selected="true"] { background-color: var(--pico-card-background-color); border-bottom-color: var(--pico-card-background-color); }
        #map-container { position: relative; width: 100%; height: 600px; background-color: #334155; border-radius: var(--pico-border-radius); overflow: hidden; cursor: grab; }
        #map-container:active { cursor: grabbing; }
        #map-pan-zoom-container { width: 100%; height: 100%; }
        #map-pan-zoom-container img { width: 100%; height: 100%; object-fit: contain; }
        .camera-icon { position: absolute; cursor: grab; display: flex; flex-direction: column; align-items: center; user-select: none; padding: 4px; border-radius: 8px; background-color: rgba(30, 41, 59, 0.7); transform-origin: top left; }
        .camera-icon.dragging { cursor: grabbing; z-index: 10; box-shadow: 0 0 15px rgba(56, 189, 248, 0.7); }
        .camera-icon-emoji { width: 30px; height: 30px; background-color: rgba(56, 189, 248, 0.8); border: 1px solid white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; }
        .camera-icon-label { color: white; font-size: 10px; margin-top: 4px; font-weight: bold; text-shadow: 1px 1px 2px black; }
        .search-results, .lpr-results { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
        .result-item { background: #334155; border-radius: var(--pico-border-radius); overflow: hidden; }
        .result-item img { width: 100%; aspect-ratio: 16/9; object-fit: cover; }
        .result-item p { padding: 0.5rem; margin: 0; font-size: 0.8rem; }
        .status-badge { font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 1rem; margin-left: 0.5rem; }
        .status-enabled { background-color: #16a34a; color: white; }
        .status-disabled { background-color: #b91c1c; color: white; }
        #camera-modal-content img { max-width: 100%; border-radius: var(--pico-border-radius); }
        #camera-modal-content ul { list-style: none; padding: 0; margin-top: 1rem; }
        #camera-modal-content li { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--pico-card-border-color); }
        #camera-modal-content li:last-child { border-bottom: none; }
        #camera-modal-content li span { color: #94a3b8; }
        .map-controls { position: absolute; top: 10px; right: 10px; z-index: 10; display: flex; flex-direction: column; gap: 5px; }
        .map-controls button { width: 40px; height: 40px; font-size: 1.5rem; padding: 0; }
        .map-layout { display: grid; grid-template-columns: 1fr 200px; gap: 1rem; }
        #camera-palette { background-color: var(--pico-form-element-background-color); border-radius: var(--pico-border-radius); padding: 1rem; height: 600px; overflow-y: auto; }
        #camera-palette h3 { margin-top: 0; font-size: 1.1rem; }
        .palette-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; border-radius: var(--pico-border-radius); cursor: grab; background-color: #475569; margin-bottom: 0.5rem; }
        .palette-item:active { cursor: grabbing; }
        .palette-item.hidden { display: none; }
    </style>
</head>
<body>
    <main class="container">
        <header style="text-align: center; margin-bottom: 2rem;">
            <h1>Frigate Intelligence Platform</h1>
        </header>

        <nav class="tabs">
            <ul>
                <li role="presentation"><a href="#" role="tab" aria-selected="true" data-tab="dashboard">Dashboard</a></li>
                <li role="presentation"><a href="#" role="tab" data-tab="map">Site Map</a></li>
                <li role="presentation"><a href="#" role="tab" data-tab="search">Explore & Search <span id="search-status-badge" class="status-badge"></span></a></li>
                <li role="presentation"><a href="#" role="tab" data-tab="lpr">LPR <span id="lpr-status-badge" class="status-badge"></span></a></li>
            </ul>
        </nav>

        <div id="dashboard-tab" class="tab-content">
            <article class="card">
                <div class="grid" style="grid-template-columns: 1.5fr 1fr 1fr auto; align-items: flex-end;">
                    <div>
                        <label for="date-range-picker">Main Date Range (for all stats)</label>
                        <input type="text" id="date-range-picker" placeholder="Select Date Range...">
                    </div>
                    <div>
                        <label for="camera-filter">Filter by Camera</label>
                        <details role="list" id="camera-filter">
                            <summary aria-haspopup="listbox" role="button" class="secondary">All Cameras</summary>
                            <ul role="listbox" id="camera-select-list" style="list-style: none; padding: 0.5rem;">
                                <li><label><input type="checkbox" id="select-all-cameras" checked> <em>Select All</em></label></li>
                            </ul>
                        </details>
                    </div>
                    <div>
                        <button id="apply-filters-btn">Apply Filters</button>
                    </div>
                    <div style="display: flex; gap: 0.5rem;">
                        <button id="export-csv-btn" class="secondary outline">Export CSV</button>
                        <button data-target="pdf-modal" onClick="toggleModal(event)" class="secondary outline">Export PDF</button>
                    </div>
                </div>
            </article>

            <div id="main-content">
                <section id="stat-panel-section" style="margin-top: 2rem;">
                    <div class="grid" style="grid-template-columns: repeat(5, 1fr);">
                        <article class="card stat-panel"><div id="stat-total-detections" class="value">-</div><div class="label">Total Detections</div></article>
                        <article class="card stat-panel"><div id="stat-anomaly" class="value">-</div><div class="label">vs. Prior Period</div></article>
                        <article class="card stat-panel"><div id="stat-most-active-cam" class="value">-</div><div class="label">Most Active Camera</div></article>
                        <article class="card stat-panel"><div id="stat-most-freq-obj" class="value">-</div><div class="label">Most Frequent Object</div></article>
                        <article class="card stat-panel"><div id="stat-busiest-hour" class="value">-</div><div class="label">Busiest Hour</div></article>
                    </div>
                </section>
                <section id="trends-section" style="margin-top: 2rem;"><article class="card"><h2>Hourly Activity Trend</h2><p id="hourly-trend-explanation" style="font-size: 0.9rem; color: #94a3b8; margin-top: -0.5rem;"><b>Note:</b> This chart's data is controlled by the <b>Main Date Range</b> filter at the top of the page.</p><div class="chart-container" style="height: 350px;"><canvas id="trends-chart"></canvas></div></article></section>
                <section id="details-section" style="margin-top: 2rem;"><div class="grid" style="grid-template-columns: 1fr 1fr 1fr; align-items: flex-start;">
                    <article class="card"><h2>Detections by Camera & Zone</h2><div id="charts-grid"></div></article>
                    <article class="card"><h2>Longest Events (Dwell Time)</h2><table id="longest-events-table"><thead><tr><th>Object</th><th>Camera</th><th>Duration</th><th></th></tr></thead><tbody></tbody></table></article>
                    <article class="card"><h2>Camera Transition Analysis</h2><table id="camera-transitions-table"><thead><tr><th>From Camera</th><th>→</th><th>To Camera</th><th>Count</th></tr></thead><tbody></tbody></table></article>
                </div></section>
            </div>
            <div id="no-data-message" style="display: none; text-align: center; padding: 2rem;" class="card"></div>
        </div>

        <div id="map-tab" class="tab-content" style="display: none;">
            <article class="card">
                <div class="grid" style="grid-template-columns: 1fr 1fr auto; align-items: flex-end;">
                    <div>
                        <label for="site-name-input">Site Name</label>
                        <input type="text" id="site-name-input" placeholder="e.g., Main Office">
                    </div>
                    <div>
                        <label for="site-address-input">Site Address</label>
                        <input type="text" id="site-address-input" placeholder="e.g., 123 Main St, Anytown">
                    </div>
                    <div>
                         <button id="save-site-info-btn">Save Site Info</button>
                    </div>
                </div>
                 <div class="grid" style="grid-template-columns: 1fr auto; margin-top: 1rem;">
                    <div>
                        <label for="map-image-upload">Upload Map Background Image</label>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <input type="file" id="map-image-upload" accept="image/*">
                            <button id="map-upload-btn" class="secondary" style="display: none;">Upload Selected Image</button>
                        </div>
                    </div>
                    <div>
                        <label>Map Controls</label>
                        <button id="reset-map-btn" class="secondary outline">Reset Positions</button>
                    </div>
                </div>
                <div class="map-layout" style="margin-top: 1rem;">
                    <div id="map-container">
                        <div id="map-pan-zoom-container">
                            <img id="map-image" src="/storage/sitemap.jpg" onerror="this.style.display='none'; this.parentElement.parentElement.querySelector('p').style.display='block';">
                        </div>
                        <p style="text-align: center; margin-top: 2rem; display: none;">No map image uploaded.</p>
                        <div class="map-controls">
                            <button id="zoom-in-btn">+</button>
                            <button id="zoom-out-btn">-</button>
                        </div>
                    </div>
                    <div id="camera-palette">
                        <h3>Available Cameras</h3>
                        <div id="palette-items"></div>
                    </div>
                </div>
            </article>
        </div>

        <div id="search-tab" class="tab-content" style="display: none;">
            <article class="card">
                <form id="search-form" class="grid" style="grid-template-columns: 1fr auto;">
                    <input type="search" id="search-input" name="query" placeholder="e.g., a red car driving down the street..." required>
                    <button type="submit">Search</button>
                </form>
                <div id="search-results" class="search-results" style="margin-top: 2rem;"></div>
            </article>
        </div>
        
        <div id="lpr-tab" class="tab-content" style="display: none;">
            <article class="card">
                <div class="grid" style="grid-template-columns: 1fr 1fr 1fr 1fr auto; align-items: flex-end; gap: 1rem;">
                    <div>
                        <label for="lpr-date-range-picker">Date Range</label>
                        <input type="text" id="lpr-date-range-picker" placeholder="Select Date Range...">
                    </div>
                    <div>
                        <label for="lpr-start-time">Start Time (Optional)</label>
                        <input type="time" id="lpr-start-time">
                    </div>
                    <div>
                        <label for="lpr-end-time">End Time (Optional)</label>
                        <input type="time" id="lpr-end-time">
                    </div>
                    <div>
                        <label for="lpr-camera-filter">Filter by Camera</label>
                        <details role="list" id="lpr-camera-filter">
                            <summary aria-haspopup="listbox" role="button" class="secondary">All Cameras</summary>
                            <ul role="listbox" id="lpr-camera-select-list" style="list-style: none; padding: 0.5rem;">
                                <li><label><input type="checkbox" id="lpr-select-all-cameras" checked> <em>Select All</em></label></li>
                            </ul>
                        </details>
                    </div>
                    <div>
                        <button id="lpr-apply-filters-btn">Apply Filters</button>
                    </div>
                </div>
            </article>
            <article class="card" style="margin-top: 2rem;">
                <h2>License Plate Recognizer Events</h2>
                <div id="lpr-results" class="lpr-results"></div>
            </article>
        </div>

    </main>
    <footer id="status-bar">Initializing...</footer>

    <dialog id="pdf-modal">
        <article>
            <header><a href="#close" aria-label="Close" class="close" data-target="pdf-modal" onClick="toggleModal(event)"></a><h3>Customize PDF Report</h3></header>
            <div class="modal-body">
                <p>Check/uncheck to include/exclude sections from the report.</p>
                <ul id="pdf-sections-list" style="list-style: none; padding: 0;">
                    <li><label><input type="checkbox" data-section="summary_panels" checked> Summary Panels</label></li>
                    <li><label><input type="checkbox" data-section="hourly_trend" checked> Hourly Trend Chart</label></li>
                    <li><label><input type="checkbox" data-section="dwell_time" checked> Dwell Time / Longest Events</label></li>
                    <li><label><input type="checkbox" data-section="camera_transitions" checked> Camera Transition Analysis</label></li>
                    <hr>
                    <li><label><input type="checkbox" data-section="site_map" checked> Site Map Snapshot</label></li>
                    <li><label><input type="checkbox" data-section="lpr_report" checked> License Plate Report</label></li>
                </ul>
            </div>
            <footer><button id="generate-pdf-btn" class="contrast">Generate PDF</button></footer>
        </article>
    </dialog>
    
    <dialog id="heatmap-modal">
        <article>
            <header><a href="#close" aria-label="Close" class="close" data-target="heatmap-modal" onClick="toggleModal(event)"></a><h3 id="heatmap-title">Activity Heatmap</h3></header>
            <div id="heatmap-container" style="position: relative; width: 100%; aspect-ratio: 16/9; background-size: contain; background-repeat: no-repeat; background-position: center;"></div>
        </article>
    </dialog>
    
    <dialog id="camera-modal">
        <article>
            <header><a href="#close" aria-label="Close" class="close" data-target="camera-modal" onClick="toggleModal(event)"></a><h3 id="camera-modal-title">Camera Status</h3></header>
            <div id="camera-modal-content"></div>
        </article>
    </dialog>

<script>
const elements = {
    tabs: document.querySelectorAll('.tabs [role="tab"]'),
    tabContents: document.querySelectorAll('.tab-content'),
    mapContainer: document.getElementById('map-container'),
    mapImage: document.getElementById('map-image'),
    mapImageUpload: document.getElementById('map-image-upload'),
    mapUploadBtn: document.getElementById('map-upload-btn'),
    mapPanZoomContainer: document.getElementById('map-pan-zoom-container'),
    zoomInBtn: document.getElementById('zoom-in-btn'),
    zoomOutBtn: document.getElementById('zoom-out-btn'),
    cameraPalette: document.getElementById('palette-items'),
    resetMapBtn: document.getElementById('reset-map-btn'),
    siteNameInput: document.getElementById('site-name-input'),
    siteAddressInput: document.getElementById('site-address-input'),
    saveSiteInfoBtn: document.getElementById('save-site-info-btn'),
    searchForm: document.getElementById('search-form'),
    searchInput: document.getElementById('search-input'),
    searchResults: document.getElementById('search-results'),
    lprResults: document.getElementById('lpr-results'),
    searchStatusBadge: document.getElementById('search-status-badge'),
    lprStatusBadge: document.getElementById('lpr-status-badge'),
    pdfSectionsList: document.getElementById('pdf-sections-list'),
    heatmapContainer: document.getElementById('heatmap-container'),
    heatmapTitle: document.getElementById('heatmap-title'),
    datePicker: document.getElementById('date-range-picker'),
    cameraSelectList: document.getElementById('camera-select-list'),
    selectAllCameras: document.getElementById('select-all-cameras'),
    applyFiltersBtn: document.getElementById('apply-filters-btn'),
    exportCsvBtn: document.getElementById('export-csv-btn'),
    exportPdfBtn: document.querySelector('[data-target="pdf-modal"]'),
    generatePdfBtn: document.getElementById('generate-pdf-btn'),
    chartsGrid: document.getElementById('charts-grid'),
    longestEventsTable: document.querySelector('#longest-events-table tbody'),
    cameraTransitionsTable: document.querySelector('#camera-transitions-table tbody'),
    statusBar: document.getElementById('status-bar'),
    noDataMessage: document.getElementById('no-data-message'),
    mainContent: document.getElementById('main-content'),
    trendsChart: document.getElementById('trends-chart'),
    hourlyTrendExplanation: document.getElementById('hourly-trend-explanation'),
    cameraFilterSummary: document.querySelector('#camera-filter summary'),
    cameraModalTitle: document.getElementById('camera-modal-title'),
    cameraModalContent: document.getElementById('camera-modal-content'),
    lprDatePicker: document.getElementById('lpr-date-range-picker'),
    lprCameraSelectList: document.getElementById('lpr-camera-select-list'),
    lprSelectAllCameras: document.getElementById('lpr-select-all-cameras'),
    lprApplyFiltersBtn: document.getElementById('lpr-apply-filters-btn'),
    lprCameraFilterSummary: document.querySelector('#lpr-camera-filter summary'),
    lprStartTime: document.getElementById('lpr-start-time'),
    lprEndTime: document.getElementById('lpr-end-time'),
    statPanels: {
        total: document.getElementById('stat-total-detections'),
        anomaly: document.getElementById('stat-anomaly'),
        cam: document.getElementById('stat-most-active-cam'),
        obj: document.getElementById('stat-most-freq-obj'),
        hour: document.getElementById('stat-busiest-hour'),
    }
};

let chartInstances = {};
const CHART_COLORS = ['#38bdf8', '#fb923c', '#4ade80', '#a78bfa', '#f472b6', '#facc15', '#2dd4bf', '#e11d48', '#f59e0b'];
let dateRangePicker;
let lprDateRangePicker;
let isModalOpen = false;
let allCameras = [];
let siteLayout = { cameras: {}, siteName: '', siteAddress: '' };
let panzoomInstance;

function updateStatus(message, isError = false) {
    console.log(`Status: ${message}`);
    elements.statusBar.textContent = message;
    elements.statusBar.style.backgroundColor = isError ? '#b91c1c' : '#475569';
}

function formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
}

function destroyAllCharts() {
    Object.values(chartInstances).forEach(chart => chart.destroy());
    chartInstances = {};
}

function renderStatPanels(stats) {
    elements.statPanels.total.textContent = stats?.total_detections ?? '0';
    elements.statPanels.cam.textContent = stats?.most_active_camera?.camera || 'N/A';
    elements.statPanels.obj.textContent = stats?.most_frequent_object?.label || 'N/A';
    elements.statPanels.hour.textContent = stats?.busiest_hour?.hour ? `${stats.busiest_hour.hour}:00` : 'N/A';
    
    const anomalyPercent = stats?.anomaly_percent ?? 0;
    const anomalyEl = elements.statPanels.anomaly;
    anomalyEl.textContent = `${anomalyPercent >= 0 ? '+' : ''}${anomalyPercent.toFixed(1)}%`;
    if (anomalyPercent > 10) anomalyEl.style.color = '#ef4444';
    else if (anomalyPercent < -10) anomalyEl.style.color = '#22c55e';
    else anomalyEl.style.color = '#38bdf8';
}

function renderTrendsChart(trendsData, prevTrendsData) {
    trendsData = trendsData ?? {};
    prevTrendsData = prevTrendsData ?? {};
    const labels = Array.from({ length: 24 }, (_, i) => i.toString().padStart(2, '0'));
    const allObjects = [...new Set(Object.values(trendsData).flatMap(Object.keys))];
    const datasets = allObjects.map((label, i) => ({
        type: 'bar', label: label, data: labels.map(hour => trendsData[hour]?.[label] || 0),
        backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
    }));

    datasets.push({
        type: 'line', label: 'Previous Period', data: labels.map(hour => prevTrendsData[hour] || 0),
        borderColor: 'rgba(220, 223, 230, 0.5)', borderWidth: 2, fill: false, pointRadius: 0,
    });

    if (chartInstances.trends) chartInstances.trends.destroy();
    chartInstances.trends = new Chart(elements.trendsChart.getContext('2d'), {
        data: { labels, datasets },
        options: {
            animation: false, responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { color: '#dce3e8' } } },
            scales: { y: { stacked: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: '#334155' } }, x: { stacked: true, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } } }
        }
    });
}

function renderZoneCharts(statsByCameraZone) {
    elements.chartsGrid.innerHTML = '';
    const camerasWithData = statsByCameraZone ? Object.keys(statsByCameraZone) : [];

    if (camerasWithData.length === 0) {
        elements.chartsGrid.innerHTML = '<p style="font-style: italic; color: #94a3b8;">No zone-specific detection data for this period.</p>';
        return;
    }

    camerasWithData.sort().forEach(camera => {
        const zones = statsByCameraZone[camera];
        const container = document.createElement('div');
        container.innerHTML = `<h3>${camera} <button class="outline secondary" style="font-size: 0.7rem; padding: 0.2rem 0.5rem; float: right;" onclick="generateHeatmap('${camera}')">Heatmap</button></h3><div class="chart-container" style="height: 150px" id="chart-cont-${camera}"></div>`;
        elements.chartsGrid.appendChild(container);
        
        const canvas = document.createElement('canvas');
        container.querySelector('.chart-container').appendChild(canvas);
        
        const allLabels = [...new Set(Object.values(zones).flatMap(Object.keys))];
        const sortedZones = Object.keys(zones).sort();
        const datasets = allLabels.map((label, i) => ({
            label: label,
            data: sortedZones.map(zone => zones[zone][label] || 0),
            backgroundColor: CHART_COLORS[i % CHART_COLORS.length]
        }));

        if (chartInstances[camera]) chartInstances[camera].destroy();
        chartInstances[camera] = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels: sortedZones, datasets },
            options: {
                animation: false, responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                plugins: { legend: { display: datasets.length > 1, labels: {color: '#dce3e8'} } },
                scales: { y: { stacked: true, ticks: { color: '#dce3e8' } }, x: { stacked: true, ticks: { color: '#dce3e8', precision: 0 } } }
            }
        });
    });
}

function renderLongestEvents(events) {
    elements.longestEventsTable.innerHTML = !events || events.length === 0 ? '<tr><td colspan="4" style="text-align: center; font-style: italic;">No events with duration found.</td></tr>' :
        events.map(event => `
            <tr>
                <td>${event.label}</td>
                <td>${event.camera}</td>
                <td>${formatDuration(event.duration)}</td>
                <td><a href="/api/frigate_proxy/api/events/${event.id}/clip.mp4" target="_blank" title="View Clip">🎬</a></td>
            </tr>
        `).join('');
}

function renderCameraTransitions(transitions) {
    elements.cameraTransitionsTable.innerHTML = !transitions || transitions.length === 0 ? '<tr><td colspan="4" style="text-align: center; font-style: italic;">No camera transition data.</td></tr>' :
        transitions.map(t => `
            <tr>
                <td>${t.from}</td>
                <td>→</td>
                <td>${t.to}</td>
                <td>${t.count}</td>
            </tr>
        `).join('');
}

function showLoadingState() {
    updateStatus('Loading data...');
    elements.applyFiltersBtn.setAttribute('aria-busy', 'true');
    elements.applyFiltersBtn.disabled = true;
    destroyAllCharts();
    elements.mainContent.style.display = 'none';
    elements.noDataMessage.style.display = 'none';
}

function getApiParams() {
    const selectedCameras = Array.from(document.querySelectorAll('#camera-select-list input[type="checkbox"]:not(#select-all-cameras):checked')).map(cb => cb.value);
    const dates = dateRangePicker.selectedDates;
    const start = dates[0] ? dates[0].getTime() / 1000 : '';
    const end = dates[1] ? (dates[1].getTime() / 1000) + 86399 : '';
    const params = new URLSearchParams({ start, end });
    if (selectedCameras.length > 0 && selectedCameras.length < allCameras.length) {
        params.append('cameras', selectedCameras.join(','));
    }
    return params;
}

async function fetchData() {
    showLoadingState();
    try {
        const response = await fetch(`/api/stats?${getApiParams().toString()}`);
        if (!response.ok) throw new Error(`Server responded with status ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        console.debug("Received data:", data);
        
        const hasData = data.stat_panels && data.stat_panels.total_detections > 0;
        elements.mainContent.style.display = hasData ? 'block' : 'none';
        elements.noDataMessage.style.display = hasData ? 'none' : 'block';

        if (hasData) {
            elements.hourlyTrendExplanation.innerHTML = `<b>Note:</b> This chart's data is controlled by the <b>Main Date Range</b> filter at the top of the page. Currently showing: ${data.date_range.start} to ${data.date_range.end}.`;
            renderStatPanels(data.stat_panels);
            renderTrendsChart(data.hourly_trends, data.prev_hourly_trends);
            renderZoneCharts(data.stats_by_camera_zone);
            renderLongestEvents(data.longest_events);
            renderCameraTransitions(data.camera_transitions);
        } else {
            elements.noDataMessage.innerHTML = '<h3>No detection data found for the selected filters.</h3>';
        }
        
        updateStatus(`Dashboard updated: ${new Date().toLocaleTimeString()}`);
    } catch (error) {
        console.error("Fetch Data Error:", error);
        updateStatus(`Error: ${error.message}`, true);
        elements.noDataMessage.innerHTML = `<h3>An error occurred</h3><p>${error.message}</p><p>Check the browser console and container logs for more details.</p>`;
        elements.noDataMessage.style.display = 'block';
    } finally {
        elements.applyFiltersBtn.removeAttribute('aria-busy');
        elements.applyFiltersBtn.disabled = false;
    }
}

function toggleModal(event) {
    event.preventDefault();
    const modalId = event.currentTarget.dataset.target;
    if (!modalId) return;
    const modal = document.getElementById(modalId);
    if (!modal) return;
    
    isModalOpen = !modal.open;
    modal.open = isModalOpen;
}

async function handleSearch(event) {
    event.preventDefault();
    const query = elements.searchInput.value.trim();
    if (!query) return;

    const submitBtn = event.target.querySelector('button');
    submitBtn.setAttribute('aria-busy', 'true');
    elements.searchResults.innerHTML = '';
    updateStatus(`Searching for "${query}"...`);

    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error(`Search failed with status ${response.status}`);
        const results = await response.json();
        if (results.error) throw new Error(results.error);

        if (results.length === 0) {
            elements.searchResults.innerHTML = '<p>No results found.</p>';
        } else {
            elements.searchResults.innerHTML = results.map(result => `
                <div class="result-item">
                    <a href="/api/frigate_proxy/api/events/${result.id}/clip.mp4" target="_blank">
                        <img src="/api/frigate_proxy/api/events/${result.id}/thumbnail.jpg" loading="lazy" alt="Event thumbnail">
                        <p>${new Date(result.start_time * 1000).toLocaleString()}<br>${result.camera}</p>
                    </a>
                </div>
            `).join('');
        }
        updateStatus('Search complete.');
    } catch (error) {
        console.error("Search Error:", error);
        updateStatus(`Search Error: ${error.message}`, true);
        elements.searchResults.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}</p>`;
    } finally {
        submitBtn.removeAttribute('aria-busy');
    }
}

function getLprApiParams() {
    const selectedCameras = Array.from(document.querySelectorAll('#lpr-camera-select-list input[type="checkbox"]:not(#lpr-select-all-cameras):checked')).map(cb => cb.value);
    const dates = lprDateRangePicker.selectedDates;

    let startTimestamp = '';
    if (dates[0]) {
        const startDate = new Date(dates[0]);
        const startTimeValue = elements.lprStartTime.value;
        if (startTimeValue) {
            const [hours, minutes] = startTimeValue.split(':');
            startDate.setHours(hours, minutes, 0, 0);
        } else {
            startDate.setHours(0, 0, 0, 0); // Start of the day
        }
        startTimestamp = startDate.getTime() / 1000;
    }

    let endTimestamp = '';
    if (dates.length > 0) {
        const endDate = new Date(dates[dates.length - 1]);
        const endTimeValue = elements.lprEndTime.value;
        if (endTimeValue) {
            const [hours, minutes] = endTimeValue.split(':');
            endDate.setHours(hours, minutes, 59, 999);
        } else {
            endDate.setHours(23, 59, 59, 999); // End of the day
        }
        endTimestamp = endDate.getTime() / 1000;
    }

    const params = new URLSearchParams({
        start: startTimestamp,
        end: endTimestamp
    });

    if (selectedCameras.length > 0 && selectedCameras.length < allCameras.length) {
        params.append('cameras', selectedCameras.join(','));
    }
    return params;
}

async function fetchLprData() {
    updateStatus('Loading LPR data...');
    elements.lprResults.innerHTML = '<progress></progress>';
    elements.lprApplyFiltersBtn.setAttribute('aria-busy', 'true');
    try {
        const response = await fetch(`/api/lpr?${getLprApiParams().toString()}`);
        if (!response.ok) throw new Error(`LPR fetch failed with status ${response.status}`);
        const results = await response.json();
        if (results.error) throw new Error(results.error);
        
        if (results.length === 0) {
            elements.lprResults.innerHTML = '<p>No license plate events found for the selected filters.</p>';
        } else {
            elements.lprResults.innerHTML = results.map(result => `
                <div class="result-item">
                    <a href="/api/frigate_proxy/api/events/${result.id}/clip.mp4" target="_blank">
                        <img src="/api/frigate_proxy/api/events/${result.id}/thumbnail.jpg" loading="lazy" alt="License plate event">
                         <p><strong>Plate: ${result.plate || 'N/A'}</strong><br>
                         ${result.camera} at ${new Date(result.start_time * 1000).toLocaleString()}</p>
                    </a>
                </div>
            `).join('');
        }
        updateStatus('LPR data loaded.');
    } catch(error) {
        console.error("LPR Fetch Error:", error);
        updateStatus(`LPR Error: ${error.message}`, true);
        elements.lprResults.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}</p>`;
    } finally {
        elements.lprApplyFiltersBtn.removeAttribute('aria-busy');
    }
}

async function initializeMap() {
    panzoomInstance = Panzoom(elements.mapPanZoomContainer, {
        maxScale: 10,
        minScale: 0.3,
        contain: 'outside',
        // This is the key fix: ignore pointer events on camera icons so the map doesn't pan.
        excludeClass: 'camera-icon'
    });
    elements.mapContainer.addEventListener('wheel', panzoomInstance.zoomWithWheel);
    elements.zoomInBtn.addEventListener('click', panzoomInstance.zoomIn);
    elements.zoomOutBtn.addEventListener('click', panzoomInstance.zoomOut);

    elements.mapPanZoomContainer.addEventListener('panzoomzoom', (event) => {
        const scale = event.detail.scale;
        document.querySelectorAll('.camera-icon').forEach(icon => {
            icon.style.transform = `scale(${1 / scale})`;
        });
    });

    await loadSiteLayout();
    populateCameraPalette();
    placeSavedCameras();
}

{% raw %}
/**
 * Transforms client (mouse) coordinates into the local coordinate system of the pan-zoom container.
 * This is crucial for accurate placement and dragging regardless of the current pan and zoom state.
 * @param {number} clientX The mouse's X coordinate relative to the viewport.
 * @param {number} clientY The mouse's Y coordinate relative to the viewport.
 * @returns {{x: number, y: number}} The coordinates within the map's pannable element.
 */
{% endraw %}
function getLocalCoord(clientX, clientY) {
    if (!panzoomInstance) return { x: clientX, y: clientY };

    const parentRect = elements.mapPanZoomContainer.getBoundingClientRect();
    const style = window.getComputedStyle(elements.mapPanZoomContainer);
    const matrix = new DOMMatrix(style.transform);
    const pt = new DOMPoint(clientX - parentRect.left, clientY - parentRect.top);
    const localPoint = matrix.inverse().transformPoint(pt);

    return { x: localPoint.x, y: localPoint.y };
}

function makeDraggable(element) {
    let isDragging = false;
    let lastPos; // Will store the last position {x, y} in local coordinates

    const onMouseDown = (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        // No need for stopPropagation() because of the excludeClass option in Panzoom
        
        isDragging = false;
        element.classList.add('dragging');

        // Store the initial mouse position in the map's local coordinates
        lastPos = getLocalCoord(e.clientX, e.clientY);

        const onMouseMove = (moveEvent) => {
            moveEvent.preventDefault();
            isDragging = true;

            const currentPos = getLocalCoord(moveEvent.clientX, moveEvent.clientY);
            const deltaX = currentPos.x - lastPos.x;
            const deltaY = currentPos.y - lastPos.y;

            const currentLeft = parseFloat(element.style.left) || 0;
            const currentTop = parseFloat(element.style.top) || 0;
            element.style.left = `${currentLeft + deltaX}px`;
            element.style.top = `${currentTop + deltaY}px`;

            lastPos = currentPos;
        };

        const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            element.classList.remove('dragging');

            if (isDragging) {
                saveCameraPosition(element.dataset.camera, element.style.left, element.style.top);
            } else {
                showCameraStatus(element.dataset.camera);
            }
        };

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    };

    element.addEventListener('mousedown', onMouseDown);
}


async function showCameraStatus(cameraName) {
    elements.cameraModalTitle.textContent = `Status for ${cameraName}`;
    elements.cameraModalContent.innerHTML = '<progress></progress>';
    toggleModal({ preventDefault: () => {}, currentTarget: { dataset: { target: 'camera-modal' } } });

    try {
        const response = await fetch(`/api/camera_status/${cameraName}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        elements.cameraModalContent.innerHTML = `
            <img src="${data.snapshot_url}" alt="Latest snapshot for ${data.name}">
            <ul>
                <li><span>Detections (24h)</span><strong>${data.detections_24h}</strong></li>
                <li><span>Last Detection</span><strong>${data.last_detection_time}</strong></li>
            </ul>
        `;
    } catch (error) {
        elements.cameraModalContent.innerHTML = `<p style="color: #ef4444;">Error fetching status: ${error.message}</p>`;
    }
}

async function saveSiteLayout() {
    siteLayout.siteName = elements.siteNameInput.value;
    siteLayout.siteAddress = elements.siteAddressInput.value;
    try {
        await fetch('/api/map/layout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(siteLayout)
        });
        updateStatus(`Site info saved.`);
    } catch (error) {
        updateStatus(`Error saving layout: ${error.message}`, true);
    }
}

function saveCameraPosition(camera, x, y) {
    const pannable = elements.mapPanZoomContainer;
    const xPercent = (parseFloat(x) / pannable.offsetWidth) * 100;
    const yPercent = (parseFloat(y) / pannable.offsetHeight) * 100;

    siteLayout.cameras[camera] = { x: `${xPercent}%`, y: `${yPercent}%` };
    saveSiteLayout(); // Save the entire layout object
}


async function loadSiteLayout() {
    try {
        const response = await fetch('/api/map/layout');
        siteLayout = await response.json();
        elements.siteNameInput.value = siteLayout.siteName || '';
        elements.siteAddressInput.value = siteLayout.siteAddress || '';
    } catch (error) {
        console.error('Could not load site layout', error);
        siteLayout = { cameras: {}, siteName: '', siteAddress: '' };
    }
}

function populateCameraPalette() {
    elements.cameraPalette.innerHTML = '';
    allCameras.forEach(cam => {
        const paletteItem = document.createElement('div');
        paletteItem.className = 'palette-item';
        paletteItem.dataset.camera = cam;
        paletteItem.draggable = true;
        paletteItem.innerHTML = `<span>📷</span> ${cam}`;
        elements.cameraPalette.appendChild(paletteItem);
        
        paletteItem.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', cam);
        });
    });
}

function placeSavedCameras() {
    Object.keys(siteLayout.cameras).forEach(cam => {
        const paletteItem = document.querySelector(`.palette-item[data-camera="${cam}"]`);
        if (paletteItem) {
            createCameraIconOnMap(cam, siteLayout.cameras[cam].x, siteLayout.cameras[cam].y);
            paletteItem.classList.add('hidden');
        }
    });
}

function createCameraIconOnMap(cameraName, x, y) {
    const icon = document.createElement('div');
    icon.className = 'camera-icon';
    icon.dataset.camera = cameraName;
    icon.innerHTML = `<div class="camera-icon-emoji">📷</div><div class="camera-icon-label">${cameraName}</div>`;
    icon.style.left = x;
    icon.style.top = y;
    
    // Set initial scale to match current zoom
    const scale = panzoomInstance.getScale();
    icon.style.transform = `scale(${1 / scale})`;

    elements.mapPanZoomContainer.appendChild(icon);
    makeDraggable(icon);
}

elements.mapContainer.addEventListener('dragover', (e) => {
    e.preventDefault(); // Allow drop
});

elements.mapContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    const cameraName = e.dataTransfer.getData('text/plain');
    const paletteItem = document.querySelector(`.palette-item[data-camera="${cameraName}"]`);
    if (!paletteItem || !panzoomInstance) return;

    const localCoords = getLocalCoord(e.clientX, e.clientY);
    const pannable = elements.mapPanZoomContainer;
    
    const xPercent = (localCoords.x / pannable.offsetWidth) * 100;
    const yPercent = (localCoords.y / pannable.offsetHeight) * 100;

    const xStr = `${xPercent}%`;
    const yStr = `${yPercent}%`;

    createCameraIconOnMap(cameraName, xStr, yStr);
    paletteItem.classList.add('hidden');
    saveCameraPosition(cameraName, xStr, yStr);
});

elements.resetMapBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to remove all cameras from the map?')) {
        siteLayout.cameras = {};
        saveSiteLayout(); // Save empty camera layout
        document.querySelectorAll('.camera-icon').forEach(icon => icon.remove());
        document.querySelectorAll('.palette-item').forEach(item => item.classList.remove('hidden'));
        updateStatus('Map positions have been reset.');
    }
});


elements.tabs.forEach(tab => {
    tab.addEventListener('click', e => {
        e.preventDefault();
        elements.tabs.forEach(t => t.removeAttribute('aria-selected'));
        tab.setAttribute('aria-selected', 'true');
        elements.tabContents.forEach(c => c.style.display = 'none');
        const activeTab = document.getElementById(`${tab.dataset.tab}-tab`);
        activeTab.style.display = 'block';

        if (tab.dataset.tab === 'lpr' && !tab.dataset.initialized) {
            fetchLprData();
            tab.dataset.initialized = 'true';
        }
        if (tab.dataset.tab === 'map' && !tab.dataset.initialized) {
            initializeMap();
            tab.dataset.initialized = 'true';
        }
    });
});

async function handleMapUpload() {
    const file = elements.mapImageUpload.files[0];
    if (!file) {
        updateStatus('No file selected to upload.', true);
        return;
    }

    elements.mapUploadBtn.setAttribute('aria-busy', 'true');
    updateStatus('Uploading map image...');
    const formData = new FormData();
    formData.append('mapImage', file);

    try {
        const response = await fetch('/api/map/image', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        if (!response.ok || result.error) throw new Error(result.error || `Upload failed with status ${response.status}`);
        
        elements.mapImage.src = result.path;
        elements.mapImage.style.display = 'block';
        elements.mapImage.parentElement.parentElement.querySelector('p').style.display = 'none';
        updateStatus('Map image uploaded successfully.');
    } catch (error) {
        console.error("Map Upload Error:", error);
        updateStatus(`Map Upload Error: ${error.message}`, true);
    } finally {
        elements.mapUploadBtn.removeAttribute('aria-busy');
        elements.mapUploadBtn.style.display = 'none';
    }
}

async function generateHeatmap(cameraName) {
    updateStatus(`Generating heatmap for ${cameraName}...`);
    const modal = document.getElementById('heatmap-modal');
    elements.heatmapTitle.textContent = `Activity Heatmap for ${cameraName}`;
    
    elements.heatmapContainer.innerHTML = '';
    
    const snapshotUrl = `/api/frigate_proxy/api/${cameraName}/latest.jpg?${new Date().getTime()}`;
    elements.heatmapContainer.style.backgroundImage = `url(${snapshotUrl})`;
    
    modal.open = true;
    isModalOpen = true;

    try {
        const response = await fetch(`/api/heatmap/${cameraName}?${getApiParams().toString()}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        const heatmapInstance = h337.create({ container: elements.heatmapContainer });
        heatmapInstance.setData({ max: 5, data: data });
        
        updateStatus('Heatmap generated.');
    } catch(error) {
        console.error("Heatmap Error:", error);
        updateStatus(`Heatmap Error: ${error.message}`, true);
        elements.heatmapContainer.innerHTML = `<p style="color: #ef4444; padding: 1rem;">Error generating heatmap: ${error.message}</p>`;
    }
}

async function initialize() {
    dateRangePicker = flatpickr(elements.datePicker, {
        mode: "range", dateFormat: "Y-m-d",
        defaultDate: [new Date(Date.now() - 29 * 24 * 60 * 60 * 1000), new Date()]
    });
    lprDateRangePicker = flatpickr(elements.lprDatePicker, {
        mode: "range", dateFormat: "Y-m-d",
        defaultDate: [new Date(Date.now() - 29 * 24 * 60 * 60 * 1000), new Date()]
    });

    try {
        const configStatusRes = await fetch('/api/config_status');
        const configStatus = await configStatusRes.json();
        const setupBadge = (badge, status) => {
            badge.textContent = status ? 'Enabled' : 'Disabled';
            badge.className = `status-badge ${status ? 'status-enabled' : 'status-disabled'}`;
        };
        setupBadge(elements.searchStatusBadge, configStatus.semantic_search);
        setupBadge(elements.lprStatusBadge, configStatus.lpr);
        if (!configStatus.semantic_search) elements.searchInput.disabled = true;

        const response = await fetch('/api/cameras');
        allCameras = await response.json();
        if (allCameras && allCameras.length > 0) {
            allCameras.forEach(camera => {
                // Populate dashboard filter
                const li = document.createElement('li');
                li.innerHTML = `<label><input type="checkbox" name="camera" value="${camera}" checked> ${camera}</label>`;
                elements.cameraSelectList.appendChild(li);
                // Populate LPR filter
                const lprLi = li.cloneNode(true);
                elements.lprCameraSelectList.appendChild(lprLi);
            });
        }
    } catch (e) {
        console.error("Failed to load cameras or config status", e);
        elements.cameraSelectList.innerHTML = '<li>Error loading cameras.</li>';
        updateStatus('Error initializing: could not fetch cameras.', true);
    }

    elements.lprSelectAllCameras.addEventListener('change', (e) => {
        document.querySelectorAll('#lpr-camera-select-list input[name="camera"]').forEach(cb => cb.checked = e.target.checked);
        updateLprCameraFilterSummary();
    });
    elements.lprCameraSelectList.addEventListener('change', (e) => {
        if (e.target.id !== 'lpr-select-all-cameras') {
            const allBoxes = document.querySelectorAll('#lpr-camera-select-list input[name="camera"]');
            elements.lprSelectAllCameras.checked = Array.from(allBoxes).every(cb => cb.checked);
        }
        updateLprCameraFilterSummary();
    });
    function updateLprCameraFilterSummary() {
        const allBoxes = document.querySelectorAll('#lpr-camera-select-list input[name="camera"]');
        const selectedCount = Array.from(allBoxes).filter(cb => cb.checked).length;
        if (selectedCount === allBoxes.length) elements.lprCameraFilterSummary.textContent = 'All Cameras';
        else if (selectedCount === 0) elements.lprCameraFilterSummary.textContent = 'No Cameras Selected';
        else elements.lprCameraFilterSummary.textContent = `${selectedCount} Camera(s) Selected`;
    }
    elements.lprApplyFiltersBtn.addEventListener('click', fetchLprData);


    elements.selectAllCameras.addEventListener('change', (e) => {
        document.querySelectorAll('#camera-select-list input[name="camera"]').forEach(cb => cb.checked = e.target.checked);
        updateCameraFilterSummary();
    });
    elements.cameraSelectList.addEventListener('change', (e) => {
        if (e.target.id !== 'select-all-cameras') {
            const allBoxes = document.querySelectorAll('#camera-select-list input[name="camera"]');
            elements.selectAllCameras.checked = Array.from(allBoxes).every(cb => cb.checked);
        }
        updateCameraFilterSummary();
    });

    function updateCameraFilterSummary() {
        const allBoxes = document.querySelectorAll('#camera-select-list input[name="camera"]');
        const selectedCount = Array.from(allBoxes).filter(cb => cb.checked).length;
        if (selectedCount === allBoxes.length) elements.cameraFilterSummary.textContent = 'All Cameras';
        else if (selectedCount === 0) elements.cameraFilterSummary.textContent = 'No Cameras Selected';
        else elements.cameraFilterSummary.textContent = `${selectedCount} Camera(s) Selected`;
    }

    elements.applyFiltersBtn.addEventListener('click', fetchData);
    elements.exportCsvBtn.addEventListener('click', () => {
        window.location.href = `/api/export/csv?${getApiParams().toString()}`;
    });
    elements.searchForm.addEventListener('submit', handleSearch);
    elements.mapImageUpload.addEventListener('change', () => {
        if (elements.mapImageUpload.files.length > 0) {
            elements.mapUploadBtn.style.display = 'inline-block';
        }
    });
    elements.mapUploadBtn.addEventListener('click', handleMapUpload);
    elements.saveSiteInfoBtn.addEventListener('click', saveSiteLayout);

    elements.generatePdfBtn.addEventListener('click', async () => {
        updateStatus('Generating PDF...', false);
        elements.generatePdfBtn.setAttribute('aria-busy', 'true');
        
        const checkedSections = {};
        document.querySelectorAll('#pdf-sections-list input:checked').forEach(el => {
            checkedSections[el.dataset.section] = true;
        });

        let siteMapImage = null;
        if (checkedSections.site_map) {
            const mapTab = document.getElementById('map-tab');
            const currentTab = document.querySelector('.tabs [role="tab"][aria-selected="true"]').dataset.tab;
            const mapTabWasActive = mapTab.style.display !== 'none';
    
            // 1. Temporarily show the map tab if it's not active
            if (!mapTabWasActive) {
                mapTab.style.display = 'block';
            }
    
            try {
                // Give the browser a moment to render the map if it was hidden
                await new Promise(resolve => setTimeout(resolve, 100)); 
                
                // 2. Capture the canvas
                const canvas = await html2canvas(elements.mapContainer, {
                    useCORS: true,
                    backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--pico-card-background-color')
                });
                siteMapImage = canvas.toDataURL('image/jpeg', 0.9);
            } catch (e) {
                console.error("Error generating map canvas:", e);
                updateStatus("Could not generate map image for PDF.", true);
            } finally {
                // 3. Restore original tab visibility
                if (!mapTabWasActive) {
                    mapTab.style.display = 'none';
                }
            }
        }

        const payload = {
            ...Object.fromEntries(getApiParams()),
            trendsChartImage: chartInstances.trends ? chartInstances.trends.toBase64Image() : null,
            sections: checkedSections,
            lprFilters: Object.fromEntries(getLprApiParams()),
            siteMapData: siteLayout,
            siteMapImage: siteMapImage
        };
        
        try {
            const response = await fetch('/api/export/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error(`PDF generation failed with status ${response.status}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url; a.download = 'frigate_executive_summary.pdf';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            updateStatus('PDF downloaded successfully.');
        } catch (error) {
            console.error("PDF Generation Error:", error);
            updateStatus(`Error: ${error.message}`, true);
        } finally {
            elements.generatePdfBtn.removeAttribute('aria-busy');
            toggleModal({ preventDefault: () => {}, currentTarget: { dataset: { target: 'pdf-modal' } } });
        }
    });

    updateCameraFilterSummary();
    updateLprCameraFilterSummary();
    fetchData();
}

document.addEventListener('DOMContentLoaded', initialize);
</script>
</body>
</html>
EOF
    success_msg "Created index.html."
}

install_addon() {
    section_header "Installing Frigate Reporter Addon"
    check_dependencies
    find_frigate_data_path

    create_addon_files

    section_header "Building Docker Image"
    info_msg "Building '${DOCKER_IMAGE_NAME}' image. This may take a moment..."
    if ! docker build --no-cache -t ${DOCKER_IMAGE_NAME} "${ADDON_DIR}"; then
        error_msg "Docker image build failed."
        exit 1
    fi
    success_msg "Docker image built successfully."

    section_header "Starting Reporter Container"
    info_msg "Stopping and removing any old reporter containers..."
    docker stop ${DOCKER_CONTAINER_NAME} &>/dev/null
    docker rm ${DOCKER_CONTAINER_NAME} &>/dev/null

    info_msg "Starting new '${DOCKER_CONTAINER_NAME}' container..."
    warn_msg "Mounting Frigate's data directory read-only to access the database."
    warn_msg "Mounting addon storage directory to persist map data."

    DATA_DIR_HOST=$(dirname "${FRIGATE_DB_PATH}")
    
    # Use 127.0.0.1 (localhost) for reliable container-to-host communication with --network=host
    HOST_IP="127.0.0.1"
    info_msg "Using Frigate Host IP: ${HOST_IP}"

    if ! docker run -d \
      --name ${DOCKER_CONTAINER_NAME} \
      --restart=unless-stopped \
      --network=host \
      -v "${DATA_DIR_HOST}:/config:ro" \
      -v "${ADDON_DIR}/certs:/app/certs:ro" \
      -v "${ADDON_DIR}/storage:/app/storage:rw" \
      -e FRIGATE_HOST_IP="${HOST_IP}" \
      ${DOCKER_IMAGE_NAME}; then
        error_msg "Failed to start the reporter container."
        info_msg "Check the container logs for errors with: docker logs ${DOCKER_CONTAINER_NAME}"
        exit 1
    fi

    info_msg "Waiting for container to initialize..."
    sleep 5

    # Check container status after starting
    if ! docker ps --filter "name=${DOCKER_CONTAINER_NAME}" --filter "status=running" | grep -q ${DOCKER_CONTAINER_NAME}; then
        error_msg "Container failed to stay running. Please check the logs:"
        info_msg "sudo docker logs ${DOCKER_CONTAINER_NAME}"
        exit 1
    fi

    IP_ADDRESS_FOR_URL=$(hostname -I | cut -d ' ' -f1)
    success_msg "🎉 Frigate Reporter Addon is now running!"
    warn_msg "You are using a self-signed certificate, so your browser will show a security warning."
    info_msg "Please accept the warning to proceed."
    info_msg "Access the web interface at: ${COLOR_BOLD}https://${IP_ADDRESS_FOR_URL}:${ADDON_PORT}${COLOR_RESET}"
}

start_container() {
    section_header "Starting Reporter Container"
    info_msg "Attempting to start the '${DOCKER_CONTAINER_NAME}' container..."
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        error_msg "Container '${DOCKER_CONTAINER_NAME}' does not exist. Please run the 'install' command first."
        exit 1
    fi
    if ! docker start ${DOCKER_CONTAINER_NAME}; then
        error_msg "Failed to start container '${DOCKER_CONTAINER_NAME}'. Check docker logs for errors."
    else
        success_msg "Container '${DOCKER_CONTAINER_NAME}' started."
        IP_ADDRESS_FOR_URL=$(hostname -I | cut -d ' ' -f1)
        info_msg "Access the web interface at: ${COLOR_BOLD}https://${IP_ADDRESS_FOR_URL}:${ADDON_PORT}${COLOR_RESET}"
    fi
}

stop_container() {
    section_header "Stopping Reporter Container"
    info_msg "Attempting to stop the '${DOCKER_CONTAINER_NAME}' container..."
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        error_msg "Container '${DOCKER_CONTAINER_NAME}' does not exist."
        exit 1
    fi
    if ! docker stop ${DOCKER_CONTAINER_NAME} >/dev/null; then
        error_msg "Failed to stop container '${DOCKER_CONTAINER_NAME}'. It might not be running."
    else
        success_msg "Container '${DOCKER_CONTAINER_NAME}' stopped."
    fi
}

delete_addon() {
    section_header "Deleting Frigate Reporter Addon"
    info_msg "Stopping and removing the container..."
    docker stop ${DOCKER_CONTAINER_NAME} &>/dev/null
    docker rm ${DOCKER_CONTAINER_NAME} &>/dev/null
    if [ $? -eq 0 ]; then success_msg "Container '${DOCKER_CONTAINER_NAME}' removed."; else warn_msg "Container not found or already removed."; fi

    info_msg "Removing the Docker image..."
    docker rmi ${DOCKER_IMAGE_NAME} &>/dev/null
    if [ $? -eq 0 ]; then success_msg "Image '${DOCKER_IMAGE_NAME}' removed."; else warn_msg "Image not found or already removed."; fi

    read -p "${COLOR_PROMPT}Delete addon directory (${ADDON_DIR})? (yes/no): ${COLOR_RESET}" delete_files
    if [[ "$delete_files" == "yes" ]]; then
        rm -rf "${ADDON_DIR}"
        success_msg "Removed addon directory."
    else
        info_msg "Addon files kept at ${ADDON_DIR}"
    fi
    success_msg "✅ Addon has been successfully uninstalled."
}

main() {
    # Ensure script is run with root/sudo privileges
    if [ "$EUID" -ne 0 ]; then
      error_msg "This script needs to be run with sudo or as root to install dependencies and interact with Docker."
      exit 1
    fi

    case "$1" in
        install)
            install_addon
            ;;
        start)
            start_container
            ;;
        stop)
            stop_container
            ;;
        delete)
            delete_addon
            ;;
        check)
            check_system
            ;;
        *)
            echo "Usage: $0 {install|start|stop|delete|check}"
            exit 1
            ;;
    esac
}

main "$@"
