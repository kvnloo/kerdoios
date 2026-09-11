# Create the GitHub repo from a machine that can push to kvnloo

This cloud token cannot create `kvnloo/kerdoios` (GitHub 403; cursor[bot] only
pushes `kvnloo/herdr`). Do not dump the plugin into herdr.

From a checkout of this tree:

```bash
cd /path/to/kerdoios
git init
git add .
git commit -m "feat: scaffold Kerdoios Hermes compute-allotment plugin

Standalone plugin. Do not merge into NousResearch/hermes-agent."
gh repo create kvnloo/kerdoios --public --source . --remote origin --push
```

Then:

```bash
cp -R . ~/.hermes/plugins/kerdoios
hermes plugins doctor ~/.hermes/plugins/kerdoios --ci
hermes plugins enable kerdoios
```

Live provider adapters stay off until you pass `--live`. They read env already
on the host. Do not paste API keys or Tailscale credentials into issues or git.
