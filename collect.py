import time
import csv
import subprocess

CSV_FILENAME = "trace_camera.csv"
TARGET_INTERVAL = 1.0

def get_all_metrics_batch():
    cmd = (
        "cat /sys/class/power_supply/battery/current_now; echo '|||'; "
        "cat /sys/class/power_supply/battery/voltage_now; echo '|||'; "
        "settings get system screen_brightness; echo '|||'; "
        "cat /proc/net/dev; echo '|||'; "
        "cat /proc/stat; echo '|||'; "
        "dumpsys power | grep mWakefulness; echo '|||'; "
        "cat /sys/class/power_supply/battery/temp" 
    )
    
    try:
        result = subprocess.run(f"adb shell \"{cmd}\"", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        raw_output = result.stdout.strip()
        parts = raw_output.split('|||')
        
        curr_str = parts[0].strip()
        volt_str = parts[1].strip()
        i_ma = abs(int(curr_str)) / 1000.0 if curr_str else 0
        v_v = int(volt_str) / 1000000.0 if volt_str else 0
        
        br_str = parts[2].strip()
        br = int(br_str) if br_str.isdigit() else 0
        
        net_str = parts[3].strip()
        rx, tx = 0, 0
        for line in net_str.split('\n'):
            if "wlan0" in line:
                d = line.replace(':', ' ').split()
                rx = int(d[1])
                tx = int(d[9])
                break
        total_net_bytes = rx + tx
        
        cpu_str = parts[4].strip()
        cpu_line = cpu_str.split('\n')[0]
        cpu_parts = cpu_line.split()
        cpu_times = [int(x) for x in cpu_parts[1:]]
        idle_ticks = cpu_times[3]
        total_ticks = sum(cpu_times)
        
        screen_str = parts[5].strip()
        s_on = 1 if "Awake" in screen_str else 0

        temp_str = parts[6].strip()
        
        if temp_str.isdigit():
            temp_c = int(temp_str) / 10.0
        else:
            temp_c = 35.0
        
        return i_ma, v_v, br, total_net_bytes, idle_ticks, total_ticks, s_on, temp_c
        
    except Exception:
        return 0, 0, 0, 0, 0, 0, 0, 0

print(f"to {CSV_FILENAME}")

with open(CSV_FILENAME, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time", "voltage_v", "current_ma", "power_w", "cpu", "s_on", "br", "net_kbps", "temp_c"])
    
    start_time = time.time()
    _, _, _, last_net_bytes, last_idle, last_total, _, last_temp_c = get_all_metrics_batch()
    time.sleep(1.0)
    
    try:
        while True:
            cycle_start = time.time()
            i_ma, v_v, br_setting, curr_net_bytes, curr_idle, curr_total, real_s_on, temp_c = get_all_metrics_batch()
            
            delta_total = curr_total - last_total
            delta_idle = curr_idle - last_idle
            cpu_load = 1.0 - (delta_idle / delta_total) if delta_total > 0 else 0.0
            last_idle, last_total = curr_idle, curr_total
            
            delta_bytes = curr_net_bytes - last_net_bytes
            if delta_bytes < 0: delta_bytes = 0
            net_kbps = (delta_bytes / 1024.0)
            last_net_bytes = curr_net_bytes
            
            final_br = br_setting if real_s_on else 0
            p_real = v_v * (i_ma / 1000.0)
            
            elapsed_time = time.time() - start_time
            
            print(f"[{elapsed_time:.2f}s] P={p_real:.2f}W | CPU={cpu_load*100:.1f}% | Scr={real_s_on}({final_br}) | Net={net_kbps:.1f}KB | Temp={temp_c:.1f}°C")
            
            writer.writerow([f"{elapsed_time:.2f}", v_v, i_ma, p_real, f"{cpu_load:.4f}", real_s_on, final_br, f"{net_kbps:.2f}", f"{temp_c:.1f}"])
            
            proc_duration = time.time() - cycle_start
            sleep_time = TARGET_INTERVAL - proc_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped")