# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import sys
from data.log_cleaners import GeneralCleaner
from lib_code.cleaners import *

transforms = {
    "conn": ConnCleaner(),
    # "dhcp": GeneralCleaner(),
    # "dns": GeneralCleaner(),
    # "http": HeaderCleaner(),
    # "dpd": GeneralCleaner(),
    # "files": GeneralCleaner(),
    # "ftp": GeneralCleaner(),
    # "irc": GeneralCleaner(),
    # "kerberos": GeneralCleaner(),
    # "mysql": GeneralCleaner(),
    # "radius": GeneralCleaner(),
    # "sip": GeneralCleaner(),
    # "smtp": GeneralCleaner(),
    # "software": GeneralCleaner(),
    # "ssh": GeneralCleaner(),
    # "ssl": GeneralCleaner(),
    # "syslog": GeneralCleaner(),
    # "tunnel": GeneralCleaner(),
    # "weird": GeneralCleaner(),
    # "x509": GeneralCleaner(),
    # "dce_rpc": GeneralCleaner(),
    # "ntlm": GeneralCleaner(),
    # "rdp": GeneralCleaner(),
    # "smb_files": GeneralCleaner(),
    # "smb_mapping": GeneralCleaner()
}