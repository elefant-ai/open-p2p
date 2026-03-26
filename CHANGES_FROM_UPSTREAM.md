# Changes from upstream (elefant-ai/open-p2p)

このリポジトリは [elefant-ai/open-p2p](https://github.com/elefant-ai/open-p2p) のフォークです。
以下に upstream からの全変更点をまとめます。

- **upstream 起点コミット:** `c92eb78` (Initial commit)
- **現在のブランチ:** `feature/tcp-transport`

---

## 1. TCP トランスポートの追加

**コミット:** `5a8d990` — Add TCP transport support alongside existing UDS
**作者:** shimizuryos (2026-03-24)

recap との通信を、従来の UDS（Unix Domain Socket）に加え TCP でも行えるようにした。
Linux マシンで推論サーバーを動かし、Windows ゲームマシンの recap と LAN 経由で接続する構成を可能にする。

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `elefant/inference/unix_socket_server.py` | `InferenceServer` 基底クラスに `transport` / `bind_host` / `bind_port` パラメータを追加。`_start_server()` で UDS/TCP を分岐 |
| `elefant/policy_model/inference.py` | CLI 引数 `--transport` / `--bind-host` / `--bind-port` を追加。環境変数 `P2P_BIND_HOST` / `P2P_BIND_PORT` に対応 |

### 追加された起動オプション

| CLI 引数 | 環境変数 | デフォルト | 説明 |
|---------|---------|-----------|------|
| `--transport` | — | `uds` | `uds` or `tcp` |
| `--bind-host` | `P2P_BIND_HOST` | `0.0.0.0` | TCP バインドアドレス |
| `--bind-port` | `P2P_BIND_PORT` | `9000` | TCP バインドポート |

---

## 2. TCP 構成の README ドキュメント追加

**コミット:** `0b62972` — Add TCP remote inference setup guide to README
**作者:** shimizuryos (2026-03-24)

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `README.md` | 「Alternative Setup: TCP Remote Inference」セクションを追加。Linux + Windows 2台構成の手順・環境変数・要件を記載 |
