"""Local, dependency-free workstation readiness report (no network)."""
import argparse, json, os, platform, shutil, subprocess, sys
from pathlib import Path


def report():
    out={"python":sys.version.split()[0],"python_executable":sys.executable,
         "platform":platform.platform(),"cpu_logical_cores":os.cpu_count(),
         "nvidia_gpu":[],"nvidia_smi_status":"not detected"}
    exe=shutil.which('nvidia-smi')
    if exe:
        try:
            completed=subprocess.run([exe,'--query-gpu=name,memory.total,driver_version',
                                      '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            if completed.returncode==0:
                out['nvidia_gpu']=[line.strip() for line in completed.stdout.splitlines() if line.strip()]
                out['nvidia_smi_status']='available'
            else:out['nvidia_smi_status']='command failed'
        except (OSError,subprocess.TimeoutExpired):
            out['nvidia_smi_status']='command failed or timed out'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='pc_hardware_report.json')
    a=p.parse_args();out=report();Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
