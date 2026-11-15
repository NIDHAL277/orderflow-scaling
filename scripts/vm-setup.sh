#!/usr/bin/env bash
# One-time set-up of an Ubuntu virtual machine (Oracle Cloud "Always Free", Arm Ampere A1)
# for the stack of docker-compose.yml. Run it from the repository folder on the machine:
#
#   bash scripts/vm-setup.sh
#
# It installs Docker Engine and the compose plugin from Docker's apt repository, opens
# ports 80 and 443 in the machine's own firewall, creates the data folder owned by the
# user of the containers and a .env from the example. Running it again changes nothing.
# Guide: docs/deploy-oracle.md.
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo apt-get install -y -q ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
    "https://download.docker.com/linux/ubuntu $codename stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -q
  sudo apt-get install -y -q docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "Docker installed: log out and in again once so that docker works without sudo."
fi

# Oracle's Ubuntu images reject every inbound connection but SSH in iptables. This
# firewall is separate from the security list of the cloud network (see the guide).
for port in 80 443; do
  rule=(-p tcp --dport "$port" -m state --state NEW -j ACCEPT)
  if ! sudo iptables -C INPUT "${rule[@]}" 2>/dev/null; then
    reject=$(sudo iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" {print $1; exit}')
    if [ -n "$reject" ]; then
      sudo iptables -I INPUT "$reject" "${rule[@]}"   # before the catch-all REJECT
    else
      sudo iptables -A INPUT "${rule[@]}"
    fi
  fi
done
if command -v netfilter-persistent >/dev/null 2>&1; then
  sudo netfilter-persistent save >/dev/null   # keep the rules after a reboot
fi

mkdir -p data
sudo chown 1000:1000 data   # user "ofs" of the image (Dockerfile)
[ -f .env ] || cp .env.example .env
echo "Done. Next: set the hostnames in .env, then follow docs/deploy-oracle.md."
