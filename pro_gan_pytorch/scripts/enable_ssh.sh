#!/bin/bash
apt-get update && apt-get install -y openssh-server > /dev/null
mkdir /var/run/sshd

PASSWORD=$(< /dev/urandom tr -dc _A-Z-a-z-0-9 | head -c12)
echo "new password: ${PASSWORD}"
echo "root:${PASSWORD}" | chpasswd
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

echo "export VISIBLE=now" >> /etc/profile

/usr/sbin/sshd -D &
