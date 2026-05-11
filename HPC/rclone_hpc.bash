#first, install rclone on local PC to setup config:
brew install rclone
brew upgrade openssl #required in case browser OAuth fails

rclone config
#follow on screen instructions to setup google drive or whichever cloud service you require
#    No remotes found, make a new one?
#    n/s/q> n

#    name> samgdrive

#    Storage> "drive"

#    #then, need to setup a project and client id/secret on Google account:
#    https://rclone.org/drive/#making-your-own-client-id
#
#    continue to follow on-screen instructions for rclone
#    client_id> <9819....>
#
#    client_secret> <GOCS....>
#
#    scope> "drive.readonly"
#
#    service_account_file>
#
#    Edit advanced config?
#    y/n> n
#
#    Use web browser to automatically authenticate rclone with remote?
#     * Say Y if the machine running rclone has a web browser you can use
#     * Say N if running rclone on a (remote) machine without web browser access
#    If not sure try Y. If Y failed, try N.
#    y/n> Y
#
#    #Then authenticate on browser on local PC
#
#    Configure this as a Shared Drive (Team Drive)?
#    y/n> n
#
#    Configuration complete.
#    Options:
#    - type: drive
#    - client_id: 9819....
#    - client_secret: GOCS....
#    - scope: drive.readonly
#    - token: {"access_token":"ya29.....","token_type":"Bearer","refresh_token":"1//0g....","expiry":"2025-09-27T17:03:09.023798+08:00","expires_in":3599}
#    - team_drive:
#
#    Keep this "samgdrive" remote?
#    y/e/d> y
#
#    Current remotes:
#
#    Name                 Type
#    ====                 ====
#    samgdrive            drive

#confirm if config file created
cat ~/.config/rclone/rclone.conf
rclone --config ~/.config/rclone/rclone.conf listremotes

#copy contents to HPC
scp ~/.config/rclone/rclone.conf user@hpcremotehostname:/path/to/username/.config/rclone/rclone.conf
#scp ~/.config/rclone/rclone.conf user@hpcremotehostname:/home/svu/username/.config/rclone/rclone.conf

#then, can login to HPC server to browse the gdrive via the name:
rclone --config ~/.config/rclone/rclone.conf ls samgdrive:
rclone --config ~/.config/rclone/rclone.conf ls samgdrive:NUS/quant

#use copy to copy
rclone --config ~/.config/rclone/rclone.conf copy samgdrive:NUS/quant/"Copy of price.csv" /$WORK/$USER/data
rclone --config ~/.config/rclone/rclone.conf copy samgdrive:NUS/quant/"Copy of bse200.csv" /$WORK/$USER/data

#can also mount on a local folder:
mkdir -p ~/gdrive_mount
rclone mount gdrive: ~/gdrive_mount

#in case need to startover rclone, can backup previous config file:
mv ~/.config/rclone/rclone.conf ~/.config/rclone/rclone.conf.bak
#or explicitly define config path
rclone --config /path/to/new-rclone.conf config




