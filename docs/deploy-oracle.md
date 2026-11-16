# Deploying on Oracle Cloud

The whole stack of `docker-compose.yml` (daily pipeline, API, dashboard and the HTTPS
proxy) runs on one Arm virtual machine of Oracle Cloud's "Always Free" tier. Within the
free limits it costs nothing. Check Oracle's current limits and rules first: the figures
below are those of the time of writing.

## 1. Account and region

The free Arm machines exist only in the account's home region, chosen at sign-up and
fixed afterwards. Choose a region outside the United States: Binance refuses US
addresses on its live API, which the optional L2 collector uses.

## 2. Machine

Console, Compute, Instances, Create instance:

| Setting | Value |
|---|---|
| Image | Canonical Ubuntu 24.04 |
| Shape | Ampere `VM.Standard.A1.Flex`, 2 OCPUs, 12 GB of memory (the free allowance is 4 OCPUs and 24 GB in total) |
| Boot volume | 100 GB (the free allowance is 200 GB of block storage in total) |
| SSH key | upload your public key, or let the console create a pair and download it |

"Out of capacity" means that no free Arm machine is available at the moment in that
availability domain: try another one, or try again later.

## 3. Ports

Two firewalls must let HTTP and HTTPS in:

1. The cloud network: Networking, Virtual cloud networks, the network of the machine,
   Security lists, the default list, Add ingress rules: source `0.0.0.0/0`, protocol
   TCP, destination ports `80,443`.
2. The machine itself: Oracle's Ubuntu images reject everything but SSH in iptables;
   `scripts/vm-setup.sh` (step 5) adds the two rules and saves them.

## 4. Hostnames

Caddy serves the API and the dashboard on two names that point to the machine's public
address, and obtains their certificates by itself. Either:

* a free DuckDNS name (duckdns.org): create `yourname` and enter the address; the
  sub-names `api.yourname.duckdns.org` and `dashboard.yourname.duckdns.org` resolve to the
  same address;
* or a domain of your own, with two `A` records.

## 5. Code and set-up

```bash
ssh ubuntu@<public-address>
git clone https://github.com/NIDHAL277/orderflow-scaling && cd orderflow-scaling
bash scripts/vm-setup.sh
exit            # log in again once, so that docker works without sudo
```

Before the repository is public, bring the code as a git bundle instead of cloning:
`git bundle create ofs.bundle main --tags` on the laptop, `scp ofs.bundle
ubuntu@<public-address>:`, then `git clone ofs.bundle orderflow-scaling` on the machine.

In `.env`, set `OFS_API_HOST` and `OFS_DASHBOARD_HOST` to the two names; optionally the
Healthchecks.io URL and the Cloudflare R2 keys of the weekly backup.

## 6. Data

The daily run continues from the derived tables, the same as those of the weekly backup
(`scripts/backup.sh`) plus the forecasts: about 8 GB, nearly all of it daily statistics.
The raw Binance files are never copied: the
pipeline downloads what it needs. From the laptop, in the repository:

```bash
cd data
tar czf - daily_stats estimates tests forecast manifest validation simulation status cache \
  | ssh ubuntu@<public-address> "tar xzf - -C orderflow-scaling/data"
ssh ubuntu@<public-address> "sudo chown -R 1000:1000 orderflow-scaling/data"
```

## 7. Start

Once a release is tagged (`v1.0.0`, say), the release workflow publishes the image on
GHCR for `linux/amd64` and `linux/arm64`. A new GHCR package can start private: make it
public in its package settings on GitHub, or run `docker login ghcr.io` on the machine.
Then:

```bash
docker compose pull && docker compose up -d
```

Before that, build it on the machine instead: `docker compose build && docker compose up -d`.

Checks:

```bash
docker compose ps                          # pipeline, api, dashboard, caddy: running
docker compose exec pipeline ofs status
curl -s https://api.yourname.duckdns.org/health
```

The dashboard is then at `https://dashboard.yourname.duckdns.org`.

## 8. Keep it running

* Healthchecks.io alerts by email when the daily ping does not arrive.
* Updates: `git pull && docker compose pull && docker compose up -d`.
* Oracle may reclaim an Always Free machine that stays idle for seven days (low use of
  processor, network and memory). At the time of writing, upgrading the account to "Pay
  As You Go" lifts that rule while the Always Free resources stay free; a budget alert of
  1 € then warns of any charge.
