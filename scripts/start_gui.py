#!/usr/bin/env python3
"""Start or safely reuse this project's loopback-only annotation GUI.

The target is Linux (h100-jackson). Only Python's standard library is needed by
this launcher; the application itself uses the project's existing .venv.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


_LOCAL_HTTP = build_opener(ProxyHandler({}))


def read_int(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def argument(argv: list[str], name: str, default: str | None = None) -> str | None:
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def process_info(pid: int) -> dict | None:
    """Read process identity without shell parsing or printing other commands."""
    directory = Path("/proc") / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        cwd = (directory / "cwd").resolve(strict=True)
        argv = [part.decode(errors="replace") for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
        tail = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if not argv or tail[0] == "Z":
            return None
        return {"pid": pid, "cwd": cwd, "argv": argv, "start_ticks": tail[19]}
    except (OSError, ValueError, IndexError):
        return None


def owned_gui(pid: int, project: Path) -> dict | None:
    info = process_info(pid)
    if not info or info["cwd"] != project:
        return None
    argv = info["argv"]
    try:
        module = argv.index("-m")
    except ValueError:
        return None
    if module + 1 >= len(argv) or argv[module + 1] != "split_cloth.cli" or "serve" not in argv[module + 2:]:
        return None
    if not Path(argv[0]).name.startswith("python"):
        return None
    requested_project = argument(argv, "--project")
    if requested_project is not None:
        value = Path(requested_project)
        if not value.is_absolute():
            value = info["cwd"] / value
        if value.resolve() != project:
            return None
    return info


def same_process(info: dict, project: Path) -> bool:
    current = owned_gui(info["pid"], project)
    return current is not None and current["start_ticks"] == info["start_ticks"]


def find_gui(project: Path, pid_file: Path) -> dict | None:
    recorded = read_int(pid_file)
    if recorded:
        info = owned_gui(recorded, project)
        if info:
            return info
        if process_info(recorded) is not None:
            raise RuntimeError("logs/gui.pid 指向其他进程；未复用或停止它。请核对该 PID 文件后重试。")
        # A process not readable by this user must not be mistaken for stale.
        try:
            os.kill(recorded, 0)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError("logs/gui.pid 指向无权核验的进程，未触碰该进程。") from exc
        else:
            raise RuntimeError("logs/gui.pid 对应进程无法确认属于本项目，未触碰该进程。")
    candidates = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            info = owned_gui(int(entry.name), project)
            if info:
                candidates.append(info)
    if len(candidates) > 1:
        raise RuntimeError("发现多个属于本项目的 GUI；为避免停止错误实例，请先核对进程并修正 logs/gui.pid。")
    return candidates[0] if candidates else None


def gui_port(info: dict) -> int:
    host = argument(info["argv"], "--host", "127.0.0.1")
    if host != "127.0.0.1":
        raise RuntimeError("已有项目 GUI 未绑定 127.0.0.1，拒绝直接复用；可用 --restart 安全重启为本地监听。")
    try:
        port = int(argument(info["argv"], "--port", "8765"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("无法核验已有 GUI 的端口。") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("已有 GUI 端口不在有效范围内。")
    return port


def healthy(port: int) -> bool:
    try:
        with _LOCAL_HTTP.open(f"http://127.0.0.1:{port}/api/runs", timeout=0.25) as response:
            value = json.load(response)
            return response.status == 200 and isinstance(value, dict) and isinstance(value.get("runs"), list)
    except (OSError, URLError, ValueError, TimeoutError):
        return False


def wait_healthy(port: int, info: dict, project: Path, seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not same_process(info, project):
            return False
        if healthy(port):
            return True
        time.sleep(0.1)
    return False


def stop_gui(info: dict, project: Path) -> None:
    """Signal only the PID whose command, cwd and Linux start time we verified."""
    if not same_process(info, project):
        raise RuntimeError("GUI 进程身份已变化；未发送停止信号。")
    os.kill(info["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 3.0
    while same_process(info, project) and time.monotonic() < deadline:
        time.sleep(0.1)
    if same_process(info, project):
        os.kill(info["pid"], signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while same_process(info, project) and time.monotonic() < deadline:
            time.sleep(0.1)
    if same_process(info, project):
        raise RuntimeError("本项目 GUI 仍未停止；未启动重复实例。")


def available_port(preferred: int | None) -> int:
    if preferred and 1 <= preferred <= 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def save_pid_port(pid_file: Path, port_file: Path, pid: int, port: int) -> None:
    for path, value in ((pid_file, pid), (port_file, port)):
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(str(value) + "\n")
        temporary.replace(path)


def result(pid: int, port: int, args: argparse.Namespace) -> dict:
    return {"pid": pid, "port": port, "url": f"http://127.0.0.1:{port}",
            "ssh_command": f"ssh -N -L{args.local_port}:127.0.0.1:{port} {shlex.quote(args.ssh_host)}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="安全启动或复用本项目的远程 GUI，仅监听 127.0.0.1")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--restart", action="store_true", help="只重启经 PID、命令和工作目录核验的本项目 GUI")
    parser.add_argument("--local-port", type=int, default=18766, help="仅用于输出 SSH 转发命令")
    parser.add_argument("--ssh-host", default="h100-jackson", help="仅用于输出 SSH 转发命令")
    args = parser.parse_args()
    if not 1 <= args.local_port <= 65535:
        parser.error("--local-port 必须为 1–65535")
    project = args.project.resolve()
    if not Path("/proc/self/cmdline").is_file():
        raise RuntimeError("启动脚本需在 Linux 数据服务器上运行，以便严格核验进程身份。")
    if not (project / "src/split_cloth/cli.py").is_file():
        raise RuntimeError(f"项目路径无效：{project}")
    python = project / ".venv/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("项目 .venv/bin/python 不可用，请先准备独立虚拟环境。")
    logs = project / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pid_file, port_file = logs / "gui.pid", logs / "gui_port.txt"
    with (logs / ".gui_start.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = find_gui(project, pid_file)
        preferred = read_int(port_file)
        if previous:
            if args.restart:
                try:
                    preferred = gui_port(previous)
                except RuntimeError:
                    pass
                stop_gui(previous, project)
            else:
                port = gui_port(previous)
                if not wait_healthy(port, previous, project):
                    raise RuntimeError("已发现本项目 GUI，但 2 秒健康检查未通过；查看 logs/gui.log，或用 --restart 重启。")
                save_pid_port(pid_file, port_file, previous["pid"], port)
                print(json.dumps(result(previous["pid"], port, args), ensure_ascii=False))
                return 0
        port = available_port(preferred)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        command = [str(python), "-m", "split_cloth.cli", "--project", str(project), "serve", "--host", "127.0.0.1", "--port", str(port)]
        with (logs / "gui.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        info = None
        deadline = time.monotonic() + 0.5
        while info is None and child.poll() is None and time.monotonic() < deadline:
            info = owned_gui(child.pid, project)
            if info is None:
                time.sleep(0.02)
        if info is None or not wait_healthy(port, info, project):
            if info and same_process(info, project):
                stop_gui(info, project)
            raise RuntimeError("GUI 未通过 2 秒健康检查，启动失败；请查看 logs/gui.log。")
        save_pid_port(pid_file, port_file, child.pid, port)
        print(json.dumps(result(child.pid, port, args), ensure_ascii=False))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
