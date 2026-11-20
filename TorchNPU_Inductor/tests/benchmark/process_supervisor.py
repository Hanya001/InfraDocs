import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time

FATAL_PATTERNS = (
    "EngineCore failed to start",
    "WorkerProc failed to start",
    "WorkerProc hit an exception",
    "Worker failed with error",
)


def process_group_exists(pgid):
    if sys.platform.startswith("linux") and os.path.isdir("/proc"):
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", encoding="utf-8") as stat_file:
                    stat = stat_file.read()
                fields = stat[stat.rfind(")") + 2:].split()
                if len(fields) >= 3 and int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                continue
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process_group(process, grace_s=30):
    pgid = process.pid
    if not process_group_exists(pgid):
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not process_group_exists(pgid):
            break
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
    if process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def read_output(stream, events):
    try:
        for line in iter(stream.readline, ""):
            events.put(("line", line))
    finally:
        events.put(("eof", ""))


def run(command, timeout_s):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    parent_pid = os.getppid()
    events = queue.Queue()
    threading.Thread(target=read_output, args=(process.stdout, events), daemon=True).start()
    received_signal = []
    eof = False

    def on_signal(signum, frame):
        received_signal.append(signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    deadline = time.monotonic() + timeout_s
    return_code = None
    reason = None
    grace_s = 30
    main_exit_at = None
    while return_code is None:
        if received_signal:
            return_code = 128 + received_signal[0]
            reason = f"received signal {received_signal[0]}"
            grace_s = 5
            break
        if os.getppid() != parent_pid:
            return_code = 125
            reason = "parent process exited"
            grace_s = 5
            break
        if time.monotonic() >= deadline:
            return_code = 124
            reason = f"total timeout after {timeout_s}s"
            break
        child_return_code = process.poll()
        if child_return_code is not None:
            if main_exit_at is None:
                main_exit_at = time.monotonic()
            if eof or time.monotonic() - main_exit_at >= 2:
                return_code = child_return_code
                if process_group_exists(process.pid):
                    reason = "main process exited with residual children"
                    grace_s = 5
                    break
                return return_code
        try:
            kind, line = events.get(timeout=0.05 if main_exit_at is not None else 0.2)
        except queue.Empty:
            continue
        if kind == "eof":
            eof = True
        else:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except BrokenPipeError:
                return_code = 125
                reason = "output pipe closed"
                grace_s = 5
                break
            matched = next((pattern for pattern in FATAL_PATTERNS if pattern in line), None)
            if matched:
                return_code = 1
                reason = f"fatal engine event: {matched}"
                grace_s = 10
                break
        if eof and process.poll() is not None:
            return_code = process.wait()
            if process_group_exists(process.pid):
                reason = "main process exited with residual children"
                grace_s = 5
                break
            return return_code
    print(f"SUPERVISOR_STOP reason={reason} pgid={process.pid}", flush=True)
    stop_process_group(process, grace_s=grace_s)
    if reason.startswith("fatal engine event") and process.returncode not in (None, 0):
        return process.returncode if process.returncode > 0 else 128 - process.returncode
    return return_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("a command is required after --")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    raise SystemExit(run(command, args.timeout))


if __name__ == "__main__":
    main()
