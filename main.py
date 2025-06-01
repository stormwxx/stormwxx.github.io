import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import json
import re

SOLAR_WIND_DATA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
MAG_DATA_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
ALERTS_URL = "https://services.swpc.noaa.gov/products/alerts.json"
PLANETARY_K_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
XRAY_DATA_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

def fetch_solar_wind_data():
    try:
        response = requests.get(SOLAR_WIND_DATA_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching plasma data: {e}")
        return None

def fetch_magnetic_data():
    try:
        response = requests.get(MAG_DATA_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching magnetic data: {e}")
        return None

def fetch_alerts_data():
    try:
        response = requests.get(ALERTS_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching alerts data: {e}")
        return None

def fetch_planetary_k_data():
    try:
        response = requests.get(PLANETARY_K_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching K-index data: {e}")
        return None

def fetch_xray_data(days=7):
    try:
        response = requests.get(XRAY_DATA_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        filtered_data = []
        for entry in data:
            entry_time = datetime.strptime(entry['time_tag'], '%Y-%m-%dT%H:%M:%SZ')
            if entry_time >= cutoff_time:
                filtered_data.append(entry)
        return filtered_data, days
    except requests.RequestException as e:
        print(f"Error fetching X-ray data: {e}")
        return None, days

def flux_to_class(flux_value):
    if flux_value is None or flux_value < 1e-8:
        return "A"
    elif flux_value < 1e-7:
        return "A"
    elif flux_value < 1e-6:
        return "B"
    elif flux_value < 1e-5:
        return "C"
    elif flux_value < 1e-4:
        return "M"
    else:
        return "X"

def parse_xray_data(data):
    flux_by_time = {}
    for entry in data:
        if entry.get('energy') == '0.1-0.8nm':
            timestamp = datetime.strptime(entry['time_tag'], '%Y-%m-%dT%H:%M:%SZ')
            flux_val = float(entry['flux']) if entry['flux'] is not None else 0
            correction_val = float(entry['electron_correction']) if entry['electron_correction'] is not None else 0
            corrected_flux = flux_val - correction_val
            flux_by_time[timestamp] = corrected_flux if corrected_flux > 0 else None

    sorted_times = sorted(flux_by_time.keys())
    timestamps = sorted_times
    max_flux = [flux_by_time[t] for t in sorted_times]
    return timestamps, max_flux

def process_alerts_data(raw_data):
    if not raw_data:
        return None, None, []

    current_k = None
    current_g = None
    latest_alert = None
    k_history = []

    for alert in raw_data:
        if isinstance(alert, dict):
            product_id = alert.get('product_id', '')
            issue_datetime = alert.get('issue_datetime', '')

            k_match = re.search(r'K0([0-9])A', product_id)
            if k_match and product_id.startswith('K0') and 'A' in product_id:
                k_value = int(k_match.group(1))
                try:
                    alert_time = pd.to_datetime(issue_datetime)
                    k_history.append({'datetime': alert_time, 'k_index': k_value})
                except:
                    pass
                if latest_alert is None or issue_datetime > latest_alert.get('issue_datetime', ''):
                    current_k = k_value
                    if k_value >= 5:
                        if k_value == 5:
                            current_g = 1
                        elif k_value == 6:
                            current_g = 2
                        elif k_value == 7:
                            current_g = 3
                        elif k_value == 8:
                            current_g = 4
                        elif k_value >= 9:
                            current_g = 5
                        else:
                            current_g = 0
                        latest_alert = alert
                        print(f"K-index: {k_value}, G-scale: {current_g}")

    return current_k, current_g, k_history

def process_k_index_data(raw_data):
    if not raw_data or len(raw_data) < 2:
        return None

    data_rows = raw_data[1:]
    if data_rows:
        latest_row = data_rows[-1]
        try:
            if len(latest_row) >= 2:
                k_value = float(latest_row[1])
                return k_value
            else:
                k_value = float(latest_row[-1])
                return k_value
        except (ValueError, IndexError):
            return None
    return None

def process_plasma_data(raw_data, days_back=6):
    if not raw_data or len(raw_data) < 2:
        return None

    data_rows = raw_data[1:]
    df = pd.DataFrame(data_rows, columns=['time_tag', 'density', 'speed', 'temperature'])
    df['datetime'] = pd.to_datetime(df['time_tag'])

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    df = df[df['datetime'] >= start_date]

    for col in ['density', 'speed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def process_magnetic_data(raw_data, days_back=6):
    if not raw_data or len(raw_data) < 2:
        return None

    data_rows = raw_data[1:]
    df = pd.DataFrame(data_rows, columns=['time_tag', 'bx_gsm', 'by_gsm', 'bz_gsm', 'lon_gsm', 'lat_gsm', 'bt'])
    df['datetime'] = pd.to_datetime(df['time_tag'])

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    df = df[df['datetime'] >= start_date]

    for col in ['bx_gsm', 'by_gsm', 'bz_gsm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def setup_common_plot_formatting(ax, days_back):
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_facecolor('white')
    ax.tick_params(colors='black')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    if days_back <= 1:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    elif days_back <= 3:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

def add_timestamp_and_source(ax):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    ax.text(0.98, 0.05, f'Plotted by stormwx\n{current_time}', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='#f5f5f5', alpha=0.9),
            ha='right', va='bottom', color='#333333', weight='bold')

def create_xray_flux_plot(timestamps, max_flux, days, timeframe_label="3 days"):
    if not timestamps or not max_flux:
        return False

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')

    valid_flux = [f for f in max_flux if f is not None]
    max_value = max(valid_flux) if valid_flux else 0

    ax.set_ylim(1e-9, 1e-3)
    y_min, y_max = ax.get_ylim()

    colors = ['#90EE90', '#FFFF99', '#FFB347', '#9370DB', '#FF4444', '#8B0000']
    levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    labels = ['A', 'B', 'C', 'M', 'X']

    for i in range(len(levels)):
        if i == 0:
            ax.axhspan(y_min, levels[i], color=colors[i], alpha=0.15, zorder=0)
        else:
            ax.axhspan(levels[i-1], levels[i], color=colors[i], alpha=0.15, zorder=0)

        if levels[i] > y_min and levels[i] < y_max:
            ax.text(0.99, levels[i], f'  {labels[i]}', transform=ax.get_yaxis_transform(),
                   fontsize=16, fontweight='bold', va='center', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i+1], alpha=0.8))

    ax.axhspan(levels[-1], y_max, color=colors[-1], alpha=0.15, zorder=0)

    valid_data = [(t, f) for t, f in zip(timestamps, max_flux) if f is not None]
    if valid_data:
        times, fluxes = zip(*valid_data)
        ax.plot(times, fluxes, color='black', linewidth=2, solid_capstyle='round', zorder=10)

    ax.set_yscale('log')
    ax.set_ylabel('X-ray Flux (W/m²)', fontsize=12, color='black')
    ax.set_xlabel('Time (UTC)', fontsize=12, color='black')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=5)
    ax.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.3, zorder=5)

    setup_common_plot_formatting(ax, days)

    ax.text(0.98, 0.95, f'GOES-18 Long X-ray Flux\nLast {timeframe_label}', transform=ax.transAxes,
            fontsize=13, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9),
            color='black')

    if max_value > 0:
        max_class = flux_to_class(max_value)
        ax.text(0.02, 0.95, f'Max: {max_class}-class\n{max_value:.2e}', transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9),
                color='black')

    add_timestamp_and_source(ax)
    plt.tight_layout()
    plt.savefig('xray_flux.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True

def create_solar_wind_speed_plot(plasma_df, days_back=6, timeframe_label="6 days"):
    if plasma_df is None or plasma_df.empty:
        return False

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')

    speed_max = plasma_df['speed'].max()

    ax.plot(plasma_df['datetime'], plasma_df['speed'], color='#0066cc', linewidth=2.5, alpha=0.9)
    ax.fill_between(plasma_df['datetime'], plasma_df['speed'], alpha=0.3, color='#0066cc')
    ax.set_ylabel('Speed (km/s)', fontsize=12, color='black')
    ax.set_xlabel('Time (UTC)', fontsize=12, color='black')

    setup_common_plot_formatting(ax, days_back)

    ax.text(0.98, 0.95, f'Solar Wind Speed\nLast {timeframe_label}', transform=ax.transAxes, 
            fontsize=13, bbox=dict(boxstyle="round,pad=0.4", facecolor='#e6f3ff', alpha=0.9),
            ha='right', va='top', color='black', weight='bold')

    ax.text(0.02, 0.95, f'Max: {speed_max:.1f} km/s', transform=ax.transAxes, 
            fontsize=13, bbox=dict(boxstyle="round,pad=0.3", facecolor='#fff2e6', alpha=0.9),
            ha='left', va='top', color='black', weight='bold')

    add_timestamp_and_source(ax)
    plt.tight_layout()
    plt.savefig("solar_wind_speed_plot.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True

def create_magnetic_field_plot(mag_df, field, color, filename, days_back=6, timeframe_label="6 days"):
    if mag_df is None or mag_df.empty or field not in mag_df.columns:
        return False

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')

    field_max = mag_df[field].max()
    field_min = mag_df[field].min()
    field_positive = mag_df[field] >= 0
    field_negative = mag_df[field] < 0

    ax.plot(mag_df['datetime'], mag_df[field], color=color, linewidth=2.5, alpha=0.9)

    neg_color = '#ff6666' if field != 'bz_gsm' else '#cc0000'
    pos_color = '#ffaa66' if field == 'by_gsm' else ('#6666ff' if field == 'bx_gsm' else '#009900')

    ax.fill_between(mag_df['datetime'], mag_df[field], 0, 
                    where=field_negative, alpha=0.3, color=neg_color, interpolate=True)
    ax.fill_between(mag_df['datetime'], mag_df[field], 0, 
                    where=field_positive, alpha=0.3, color=pos_color, interpolate=True)

    field_label = field.replace('_gsm', '').upper()
    ax.set_ylabel(f'{field_label} (nT)', fontsize=12, color='black')
    ax.set_xlabel('Time (UTC)', fontsize=12, color='black')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)

    setup_common_plot_formatting(ax, days_back)

    bg_colors = {'bz_gsm': '#ffe6e6', 'bx_gsm': '#e6e6ff', 'by_gsm': '#fff4e6'}
    info_colors = {'bz_gsm': '#e6ffe6', 'bx_gsm': '#f0f0ff', 'by_gsm': '#fff8e6'}

    ax.text(0.98, 0.95, f'Solar Wind {field_label}\nLast {timeframe_label}', transform=ax.transAxes, 
            fontsize=13, bbox=dict(boxstyle="round,pad=0.4", facecolor=bg_colors[field], alpha=0.9),
            ha='right', va='top', color='black', weight='bold')

    ax.text(0.02, 0.95, f'Max: {field_max:.1f} nT\nMin: {field_min:.1f} nT', transform=ax.transAxes, 
            fontsize=13, bbox=dict(boxstyle="round,pad=0.3", facecolor=info_colors[field], alpha=0.9),
            ha='left', va='top', color='black', weight='bold')

    add_timestamp_and_source(ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True

def create_bz_plot(mag_df, days_back=6, timeframe_label="6 days"):
    return create_magnetic_field_plot(mag_df, 'bz_gsm', '#cc0000', 'bz_plot.png', days_back, timeframe_label)

def create_bx_plot(mag_df, days_back=6, timeframe_label="6 days"):
    return create_magnetic_field_plot(mag_df, 'bx_gsm', '#0066ff', 'bx_plot.png', days_back, timeframe_label)

def create_by_plot(mag_df, days_back=6, timeframe_label="6 days"):
    return create_magnetic_field_plot(mag_df, 'by_gsm', '#ff8800', 'by_plot.png', days_back, timeframe_label)

def create_density_plot(df, days_back=6, timeframe_label="6 days"):
    if df is None or df.empty:
        return False

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')

    density_max = df['density'].max()

    ax.plot(df['datetime'], df['density'], color='#ff6600', linewidth=2.5, alpha=0.9)
    ax.fill_between(df['datetime'], df['density'], alpha=0.3, color='#ff6600')
    ax.set_ylabel('Density (cm⁻³)', fontsize=12, color='black')
    ax.set_xlabel('Time (UTC)', fontsize=12, color='black')

    setup_common_plot_formatting(ax, days_back)

    ax.text(0.98, 0.95, f'Solar Wind Density\nLast {timeframe_label}', transform=ax.transAxes, 
            fontsize=13, bbox=dict(boxstyle="round,pad=0.4", facecolor='#fff0e6', alpha=0.9),
            ha='right', va='top', color='black', weight='bold')

    ax.text(0.02, 0.95, f'Max: {density_max:.2f} cm⁻³', transform=ax.transAxes, 
            fontsize=13, bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffebe6', alpha=0.9),
            ha='left', va='top', color='black', weight='bold')

    add_timestamp_and_source(ax)
    plt.tight_layout()
    plt.savefig("density_plot.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True

def create_dashboard_plot(plasma_df, mag_df, k_value=None, g_value=None, days_back=6, timeframe_label="6 days"):
    if plasma_df is None or plasma_df.empty:
        return False

    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Space Weather Dashboard - Last {timeframe_label}', fontsize=18, fontweight='bold', y=0.95)

    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    speed_max = plasma_df['speed'].max()
    ax1.plot(plasma_df['datetime'], plasma_df['speed'], color='#0066cc', linewidth=2, alpha=0.9)
    ax1.fill_between(plasma_df['datetime'], plasma_df['speed'], alpha=0.3, color='#0066cc')
    ax1.set_ylabel('Speed (km/s)', fontsize=11, fontweight='bold')
    ax1.set_title('Solar Wind', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, f'Max: {speed_max:.1f} km/s', transform=ax1.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='#e6f3ff', alpha=0.9),
            ha='left', va='top', fontweight='bold')

    density_max = plasma_df['density'].max()
    ax2.plot(plasma_df['datetime'], plasma_df['density'], color='#ff6600', linewidth=2, alpha=0.9)
    ax2.fill_between(plasma_df['datetime'], plasma_df['density'], alpha=0.3, color='#ff6600')
    ax2.set_ylabel('Density (cm⁻³)', fontsize=11, fontweight='bold')
    ax2.set_title('Density', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, f'Max: {density_max:.2f} cm⁻³', transform=ax2.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='#fff0e6', alpha=0.9),
            ha='left', va='top', fontweight='bold')

    if mag_df is not None and not mag_df.empty:
        field_configs = [
            ('bz_gsm', ax3, 'BZ', '#cc0000', '#009900', '#ffe6e6'),
            ('bx_gsm', ax4, 'BX', '#0066ff', '#6666ff', '#e6e6ff'),
            ('by_gsm', ax5, 'BY', '#ff8800', '#ffaa66', '#fff4e6')
        ]

        for field, ax, title, line_color, pos_color, bg_color in field_configs:
            if field in mag_df.columns:
                field_max = mag_df[field].max()
                field_min = mag_df[field].min()
                field_positive = mag_df[field] >= 0
                field_negative = mag_df[field] < 0

                ax.plot(mag_df['datetime'], mag_df[field], color=line_color, linewidth=2, alpha=0.9)
                ax.fill_between(mag_df['datetime'], mag_df[field], 0, 
                                where=field_negative, alpha=0.3, color='#ff6666', interpolate=True)
                ax.fill_between(mag_df['datetime'], mag_df[field], 0, 
                                where=field_positive, alpha=0.3, color=pos_color, interpolate=True)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
                ax.set_ylabel(f'{title} (nT)', fontsize=11, fontweight='bold')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.95, f'Max: {field_max:.1f} nT\nMin: {field_min:.1f} nT', transform=ax.transAxes, 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.9),
                        ha='left', va='top', fontweight='bold')
    else:
        for ax, title in [(ax3, 'BZ'), (ax4, 'BX'), (ax5, 'BY')]:
            ax.text(0.5, 0.5, f'No {title} Data Available', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')

    ax6.axis('off')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

    if k_value is not None:
        k_colors = {0: '#00ff00', 1: '#66ff00', 2: '#ccff00', 3: '#ffff00', 
                    4: '#ffcc00', 5: '#00ff00', 6: '#ffff00', 7: '#ff9900', 
                    8: '#ff0000', 9: '#800080'}
        displayed_k = round(k_value, 1)
        k_color = k_colors.get(int(displayed_k), '#888888')
        ax6.text(0.25, 0.8, 'K-INDEX', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax6.text(0.25, 0.5, f'{displayed_k}', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=20, fontweight='bold', color=k_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=k_color, linewidth=2))
    else:
        ax6.text(0.25, 0.65, 'K-INDEX\nN/A', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12, fontweight='bold', color='#888888')

    if g_value is not None and g_value > 0:
        g_colors = {1: '#00ff00', 2: '#ffff00', 3: '#ff9900', 4: '#ff0000', 5: '#800080'}
        g_color = g_colors.get(g_value, '#888888')
        ax6.text(0.75, 0.8, 'G-SCALE', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax6.text(0.75, 0.5, f'G{g_value}', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=20, fontweight='bold', color=g_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=g_color, linewidth=2))
    else:
        ax6.text(0.75, 0.65, 'G-SCALE\nQuiet', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12, fontweight='bold', color='#00ff00')

    ax6.axvline(x=0.5, ymin=0.1, ymax=0.9, color='gray', linestyle='--', alpha=0.5)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        setup_common_plot_formatting(ax, days_back)
        ax.set_facecolor('white')

    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels([])

    ax4.set_xlabel('Time (UTC)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (UTC)', fontsize=11, fontweight='bold')

    current_time = datetime.now().strftime('%H:%M:%S UTC')
    ax6.text(0.5, 0.2, f'Updated: {current_time}', transform=ax6.transAxes, 
            ha='center', va='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#f0f0f0', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig("solar_wind_dashboard.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return True

def update_plots(days_back, timeframe_label):
    print(f"Fetching data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    raw_plasma_data = fetch_solar_wind_data()
    raw_mag_data = fetch_magnetic_data()
    raw_alerts_data = fetch_alerts_data()
    raw_k_data = fetch_planetary_k_data()
    raw_xray_data, _ = fetch_xray_data(days_back)

    if not raw_plasma_data:
        print("Failed to fetch plasma data.")
        return False

    plasma_df = process_plasma_data(raw_plasma_data, days_back)
    mag_df = process_magnetic_data(raw_mag_data, days_back)

    alert_k, alert_g, k_history = process_alerts_data(raw_alerts_data)
    planetary_k = process_k_index_data(raw_k_data)

    current_k = alert_k if alert_k is not None else planetary_k
    current_g = alert_g

    if plasma_df is None or plasma_df.empty:
        print("No plasma data available for the specified timeframe.")
        return False

    plasma_points = len(plasma_df)
    mag_points = len(mag_df) if mag_df is not None else 0

    speed_plotted = create_solar_wind_speed_plot(plasma_df, days_back, timeframe_label)
    density_plotted = create_density_plot(plasma_df, days_back, timeframe_label)
    dashboard_plotted = create_dashboard_plot(plasma_df, mag_df, current_k, current_g, days_back, timeframe_label)

    bz_plotted = False
    bx_plotted = False
    by_plotted = False
    if mag_df is not None and not mag_df.empty:
        bz_plotted = create_bz_plot(mag_df, days_back, timeframe_label)
        bx_plotted = create_bx_plot(mag_df, days_back, timeframe_label)
        by_plotted = create_by_plot(mag_df, days_back, timeframe_label)

    xray_plotted = False
    if raw_xray_data:
        timestamps, max_flux = parse_xray_data(raw_xray_data)
        xray_plotted = create_xray_flux_plot(timestamps, max_flux, days_back, timeframe_label)
        xray_points = len(raw_xray_data)
    else:
        xray_points = 0

    print(f"Solar Wind Speed plot: {'Success' if speed_plotted else 'Failed'} ({plasma_points} points)")
    print(f"Solar Wind Density plot: {'Success' if density_plotted else 'Failed'} ({plasma_points} points)")
    print(f"Space Weather Dashboard: {'Success' if dashboard_plotted else 'Failed'} (K={current_k}, G={current_g})")
    if mag_points > 0:
        print(f"BZ plot: {'Success' if bz_plotted else 'Failed'} ({mag_points} points)")
        print(f"BX plot: {'Success' if bx_plotted else 'Failed'} ({mag_points} points)")
        print(f"BY plot: {'Success' if by_plotted else 'Failed'} ({mag_points} points)")
    else:
        print("No magnetic field data available")

    print(f"X-ray Flux plot: {'Success' if xray_plotted else 'Failed'} ({xray_points} points)")

    return True

def get_timeframe_selection():
    print("\nSelect time range for Space Weather monitoring:")
    options = [
        ("1", 0.04167, "1 hour"),
        ("2", 0.08333, "2 hours"),
        ("3", 0.25, "6 hours"),
        ("4", 0.5, "12 hours"),
        ("5", 1.0, "24 hours"),
        ("6", 2, "2 days"),
        ("7", 3, "3 days (default)"),
        ("8", 4, "4 days"),
        ("9", 5, "5 days"),
        ("10", 6, "6 days"),
        ("11", 7, "7 days")
    ]

    for i, (num, _, label) in enumerate(options, 1):
        print(f"{i}. Last {label}")

    while True:
        try:
            choice = input("\nEnter your choice (1-11, or press Enter for default): ").strip()

            if choice == '' or choice == '7':
                return 3, "3 days"

            choice_map = {num: (days, label) for num, days, label in options}
            if choice in choice_map:
                return choice_map[choice]
            else:
                print("Invalid choice. Please enter 1-11.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

def main():
    print("Space Weather Monitor")
    print("=" * 40)
    print("Monitoring: Solar Wind + X-ray Flux")

    days_back, timeframe_label = get_timeframe_selection()

    print(f"\nStarting space weather monitoring for last {timeframe_label}...")
    print("Updating every 20 seconds. Press Ctrl+C to stop.")
    print("\nPlots generated:")
    plot_files = [
        "solar_wind_speed_plot.png",
        "density_plot.png", 
        "bz_plot.png",
        "bx_plot.png",
        "by_plot.png",
        "solar_wind_dashboard.png",
        "xray_flux.png"
    ]
    for plot_file in plot_files:
        print(f"- {plot_file}")

    try:
        while True:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{current_time}] Updating plots...")

            success = update_plots(days_back, timeframe_label)
            if not success:
                print("Update failed, retrying in 20 seconds...")

            time.sleep(20)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == '__main__':
    main()
