# !/bin/python
import hashlib
import os
import socket
import traceback

import colors
import argparse
import threading
import getpass
import paramiko
import sys


def ssh_connect(username, password, hostname):
    """
    Create a new ssh connection.
    :param username:
    :param password:
    :param hostname:
    :return:
    """
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)
    return ssh


def ping(hostname):
    return True
    response = os.system("ping -c 1 " + hostname)
    # and then check the response...
    return response == 0


def concurrent_map(func, inputs):
    """
    Like map(), but invoke on different threads.
    Wait for all threads to complete.
    :param func:
    :param inputs:
    :return:
    """
    threads = [threading.Thread(target=func, args=i) for i in inputs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def get_username_password():
    username = raw_input("Username:")
    password = getpass.getpass(prompt='Password: ')
    return username, password


def execute_commands_on_server(server, commands_list,
                               username, password):
    print server
    try:
        ssh = ssh_connect(username, password, server)
        print 'executing', commands_list
        print 'on ', server
        for cmd in commands_list:
            sin, sout, serr = ssh.exec_command(cmd, get_pty=True)
            print sout.read(100)
            # print serr.read(4)
        ssh.close()
    except Exception as e:
        traceback.print_exc(e)


def execute_commands(servers_names, commands_lists):

    """
    Assigns the given lists of commands to servers from the
    given list.
    :param servers_names:
    :param commands_lists: lists of commands.
    :return:
    """
    username, password = get_username_password()

    good_free_servers = [server for server in servers_names if ping(server)]
    inputs = [(server, commands, username, password) for server, commands in zip(good_free_servers,
                                                                                 commands_lists)]
    concurrent_map(execute_commands_on_server, inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch bash scripts.')
    parser.add_argument("--servers_names", nargs="+", help="server name. server-XX should exist.",
                        required=True)
    parser.add_argument("--scripts", nargs="+", help="scripts to execute.",
                        required=True)
    parser.add_argument("--kill", type=bool, help="Kill all.",
                        required=False)
    parser.add_argument("--user", type=str, help="Username.",
                        required=False)
    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise

    if args.user:
        username = args.user
    else:
        username = os.environ.get('USER', os.environ.get('USERNAME', None))
    if args.kill:
        commands = [["killall -u {}".format(username)] for script in args.scripts]
    else:
        commands = [[script] for script in args.scripts]
    execute_commands(args.servers_names, commands)
