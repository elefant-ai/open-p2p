#!/usr/bin/env python3
"""
Simple client to test the Recap server functionality.

This script connects to the Recap server and sends commands to control the application.
"""

import argparse
import json
import shlex
import sys
from typing import Any, Dict, List

import requests


class RecapClient:
    """Client for connecting to and controlling the Recap server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.socket = None

    def send_command(self, command: Dict[str, Any] | str) -> Dict[str, Any]:
        """Send a command to the server and return the response."""
        if isinstance(command, str):
            command = {command: None}

        try:
            response = requests.post(
                f"http://{self.host}:{self.port}/command",
                json=command,
            )

            if response.status_code != 200:
                return {
                    "Error": {
                        "error": f"Server returned status code {response.status_code}"
                    }
                }

            return response.json()
        except Exception as e:
            return {"Error": {"error": f"Communication error: {e}"}}

    def refresh_devices(self) -> Dict[str, Any]:
        """Refresh the device list."""
        return self.send_command("Refresh")

    def list_targets(self) -> Dict[str, Any]:
        """List available target windows."""
        return self.send_command("ListTargets")

    def set_target(self, title: str) -> Dict[str, Any]:
        """Set the target window."""
        return self.send_command({"SetTarget": {"title": title}})

    def set_task(self, task: str) -> Dict[str, Any]:
        """Set the task name."""
        return self.send_command({"SetTask": {"task": task}})

    def set_env(self, env: str) -> Dict[str, Any]:
        """Set the environment."""
        return self.send_command({"SetEnv": {"env": env}})

    def set_env_subtype(self, env_subtype: str) -> Dict[str, Any]:
        """Set the environment subtype."""
        return self.send_command({"SetEnvSubtype": {"env_subtype": env_subtype}})

    def set_user(self, user: str) -> Dict[str, Any]:
        """Set the user."""
        return self.send_command({"SetUser": {"user": user}})

    def save_settings(self) -> Dict[str, Any]:
        """Save current settings."""
        return self.send_command("SaveSettings")

    def toggle_recording(self) -> Dict[str, Any]:
        """Toggle recording on/off."""
        return self.send_command("ToggleRecording")

    def toggle_recording_with_inference(self) -> Dict[str, Any]:
        """Toggle recording with inference on/off."""
        return self.send_command("ToggleRecordingWithInference")

    def toggle_playback(self) -> Dict[str, Any]:
        """Toggle playback on/off."""
        return self.send_command("TogglePlayback")

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return self.send_command("GetStatus")

    def exit_app(self) -> Dict[str, Any]:
        """Exit the application."""
        return self.send_command("Exit")

    def set_window_size(self, width: int, height: int) -> Dict[str, Any]:
        """Set the window size."""
        return self.send_command({"SetWindowSize": {"width": width, "height": height}})

    def get_window_size(self) -> Dict[str, Any]:
        """Get the current window size."""
        return self.send_command("GetWindowSize")

    def set_window_position(self, x: int, y: int) -> Dict[str, Any]:
        """Set the window position."""
        return self.send_command({"SetWindowPosition": {"x": x, "y": y}})

    def get_window_position(self) -> Dict[str, Any]:
        """Get the current window position."""
        return self.send_command("GetWindowPosition")

    def move_mouse(self, x: float, y: float) -> Dict[str, Any]:
        """Move mouse to absolute position."""
        return self.send_command({"MoveMouse": {"x": x, "y": y}})

    def playback_file(self, file: str) -> Dict[str, Any]:
        """Start playback from a file."""
        return self.send_command({"Playback": {"path": file}})

    def toggle_model_control(self) -> Dict[str, Any]:
        """Toggle model control on/off."""
        return self.send_command("ToggleModelControl")

    def start_program(self, program: str, args: List[str]) -> Dict[str, Any]:
        """Start a program by name."""
        return self.send_command({"StartProgram": {"name": program, "args": args}})


def print_response(response: Dict[str, Any]):
    """Pretty print a server response."""
    print(f"Response: {json.dumps(response, indent=2)}")


def print_usage_instructions():
    """Print detailed usage instructions."""
    print("\nRecap Server Client - Usage Instructions")
    print("=" * 50)
    print("\nConnection:")
    print("  The client connects to the Recap server (default: 127.0.0.1:8080)")
    print("  Make sure the Recap application is running before using this client.")
    print("\nUsage Modes:")
    print("  1. Interactive Mode:")
    print("     python client.py")
    print("     - Starts an interactive session where you can type commands")
    print("     - Type 'quit' to exit the interactive session")
    print("\n  2. Single Command Mode:")
    print("     python client.py --command <cmd> [--args <arg1> <arg2> ...]")
    print("     - Executes a single command and exits")
    print("\nAvailable Commands:")
    print("  System Commands:")
    print("    status                    - Get application status")
    print("    refresh                   - Refresh device list")
    print("    save                      - Save current settings")
    print("    exit                      - Exit Recap application")
    print("\n  Target Management:")
    print("    list                      - List available target windows")
    print("    target <title>            - Set target window by title")
    print("\n  Configuration:")
    print("    task <name>               - Set task name")
    print("    env <environment>         - Set environment")
    print("    env_subtype <subtype>     - Set environment subtype")
    print("    user <username>           - Set user")
    print("\n  Recording Control:")
    print("    record                    - Toggle recording on/off")
    print("    record_inf                - Toggle recording with inference")
    print("    playback                  - Toggle playback")
    print("    playback_file <file>      - Start playback from a file")
    print("    model_control             - Toggle model control on/off")
    print("\n  Program Control:")
    print("    start_program <program> [args...] - Start a program by name")
    print("                                        Use quotes for names with spaces")
    print("\n  Window Control:")
    print("    size <width> <height>     - Set window size (integers)")
    print("    get_size                  - Get current window size")
    print("    position <x> <y>          - Set window position (integers)")
    print("    get_position              - Get current window position")
    print("\n  Mouse Control:")
    print("    mouse <x> <y>             - Move mouse to position (floats)")
    print("\nExamples:")
    print("  python client.py --command status")
    print("  python client.py --command target --args Calculator")
    print("  python client.py --command start_program --args notepad.exe")
    print(
        '  python client.py --command start_program --args "Visual Studio Code" file.txt'
    )
    print("  python client.py --command model_control")
    print("  python client.py --command size --args 1920 1080")
    print("  python client.py --command position --args 100 100")
    print("  python client.py --command mouse --args 500.5 300.2")
    print("\nOptions:")
    print("  --host <host>             - Server host (default: 127.0.0.1)")
    print("  --port <port>             - Server port (default: 8080)")
    print("  --command <cmd>           - Command to execute")
    print("  --args <arg1> <arg2> ...  - Arguments for the command")
    print("  --help, -h                - Show this help message")
    print("")


def print_interactive_mode_help():
    """Run the client in interactive mode."""
    print("\nRecap Client Interactive Mode")
    print("Available commands:")
    print("  help - Show this help message")
    print("  refresh - Refresh device list")
    print("  list - List available target windows")
    print("  target <title> - Set target window")
    print("  task <name> - Set task name")
    print("  env <environment> - Set environment")
    print("  env_subtype <subtype> - Set environment subtype")
    print("  user <username> - Set user")
    print("  save - Save settings")
    print(
        "  start_program <program> [args...] - Start a program by name (use quotes for names with spaces)"
    )
    print("  record - Toggle recording")
    print("  record_inf - Toggle recording with inference")
    print("  playback - Toggle playback")
    print("  playback_file <file> - Start playback from a file")
    print("  model_control - Toggle model control")
    print("  status - Get current status")
    print("  size <width> <height> - Set window size")
    print("  get_size - Get current window size")
    print("  position <x> <y> - Set window position")
    print("  get_position - Get current window position")
    print("  mouse <x> <y> - Move mouse to position")
    print("  exit - Exit application")
    print("  quit - Exit this client")
    print("\nExamples:")
    print('  start_program "Visual Studio Code" myfile.txt')
    print('  start_program "Google Chrome" --new-window https://example.com')
    print()


def interactive_mode(client: RecapClient):
    print_interactive_mode_help()
    while True:
        try:
            user_input = input("recap> ").strip()
            if not user_input:
                continue

            try:
                parts = shlex.split(user_input)
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

            if not parts:
                continue

            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []

            if cmd == "quit":
                break
            elif cmd == "help":
                print_interactive_mode_help()
                continue
            elif cmd == "refresh":
                response = client.refresh_devices()
            elif cmd == "list":
                response = client.list_targets()
            elif cmd == "target" and args:
                response = client.set_target(" ".join(args))
            elif cmd == "task" and args:
                response = client.set_task(" ".join(args))
            elif cmd == "env" and args:
                response = client.set_env(" ".join(args))
            elif cmd == "env_subtype" and args:
                response = client.set_env_subtype(" ".join(args))
            elif cmd == "user" and args:
                response = client.set_user(" ".join(args))
            elif cmd == "save":
                response = client.save_settings()
            elif cmd == "start_program" and args:
                name = args[0]
                cli_args = args[1:] if len(args) > 1 else []
                response = client.start_program(name, cli_args)
            elif cmd == "record":
                response = client.toggle_recording()
            elif cmd == "record_inf":
                response = client.toggle_recording_with_inference()
            elif cmd == "playback":
                response = client.toggle_playback()
            elif cmd == "model_control":
                response = client.toggle_model_control()
            elif cmd == "status":
                response = client.get_status()
            elif cmd == "size" and len(args) >= 2:
                try:
                    width = int(args[0])
                    height = int(args[1])
                    response = client.set_window_size(width, height)
                except ValueError:
                    print("Width and height must be integers")
                    continue
            elif cmd == "get_size":
                response = client.get_window_size()
            elif cmd == "position" and len(args) >= 2:
                try:
                    x = int(args[0])
                    y = int(args[1])
                    response = client.set_window_position(x, y)
                except ValueError:
                    print("X and Y coordinates must be integers")
                    continue
            elif cmd == "get_position":
                response = client.get_window_position()
            elif cmd == "mouse" and len(args) >= 2:
                try:
                    x = float(args[0])
                    y = float(args[1])
                    response = client.move_mouse(x, y)
                except ValueError:
                    print("X and Y coordinates must be numbers")
                    continue
            elif cmd == "playback_file":
                if not args or len(args) < 1 or len(args) > 1:
                    print("Usage: playback_file <file>")
                    continue
                response = client.playback_file(args[0])
            elif cmd == "exit":
                response = client.exit_app()
                print_response(response)
                break
            else:
                print(f"Unknown command: {cmd}")
                continue

            print_response(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break


def main():
    # Check for help flag first
    if len(sys.argv) > 1 and (sys.argv[1] in ["-h", "--help", "help"]):
        print_usage_instructions()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Recap Server Client",
        add_help=False,  # Disable default help to use our custom help
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Server port (default: 8080)"
    )
    parser.add_argument("--command", help="Single command to execute")
    parser.add_argument("--args", nargs="*", help="Arguments for the command")

    try:
        args = parser.parse_args()
    except SystemExit:
        print_usage_instructions()
        sys.exit(1)

    client = RecapClient(args.host, args.port)

    try:
        if args.command:
            # Single command mode
            cmd = args.command.lower()
            cmd_args = args.args or []

            if cmd == "refresh":
                response = client.refresh_devices()
            elif cmd == "list":
                response = client.list_targets()
            elif cmd == "target" and cmd_args:
                response = client.set_target(" ".join(cmd_args))
            elif cmd == "task" and cmd_args:
                response = client.set_task(" ".join(cmd_args))
            elif cmd == "env" and cmd_args:
                response = client.set_env(" ".join(cmd_args))
            elif cmd == "env_subtype" and cmd_args:
                response = client.set_env_subtype(" ".join(cmd_args))
            elif cmd == "user" and cmd_args:
                response = client.set_user(" ".join(cmd_args))
            elif cmd == "save":
                response = client.save_settings()
            elif cmd == "start_program" and cmd_args:
                name = cmd_args[0]
                args = cmd_args[1:] if len(cmd_args) > 1 else []
                response = client.start_program(name, args)
            elif cmd == "record":
                response = client.toggle_recording()
            elif cmd == "record_inf":
                response = client.toggle_recording_with_inference()
            elif cmd == "playback":
                response = client.toggle_playback()
            elif cmd == "model_control":
                response = client.toggle_model_control()
            elif cmd == "status":
                response = client.get_status()
            elif cmd == "size" and len(cmd_args) >= 2:
                try:
                    width = int(cmd_args[0])
                    height = int(cmd_args[1])
                    response = client.set_window_size(width, height)
                except ValueError:
                    print("Width and height must be integers")
                    print_usage_instructions()
                    sys.exit(1)
            elif cmd == "get_size":
                response = client.get_window_size()
            elif cmd == "position" and len(cmd_args) >= 2:
                try:
                    x = int(cmd_args[0])
                    y = int(cmd_args[1])
                    response = client.set_window_position(x, y)
                except ValueError:
                    print("X and Y coordinates must be integers")
                    print_usage_instructions()
                    sys.exit(1)
            elif cmd == "get_position":
                response = client.get_window_position()
            elif cmd == "mouse" and len(cmd_args) >= 2:
                try:
                    x = float(cmd_args[0])
                    y = float(cmd_args[1])
                    response = client.move_mouse(x, y)
                except ValueError:
                    print("X and Y coordinates must be numbers")
                    print_usage_instructions()
                    sys.exit(1)
            elif cmd == "exit":
                response = client.exit_app()
            else:
                print(f"Unknown command or missing arguments: {cmd}")
                print_usage_instructions()
                sys.exit(1)

            print_response(response)
        else:
            # Interactive mode
            interactive_mode(client)

    finally:
        ...


if __name__ == "__main__":
    main()
