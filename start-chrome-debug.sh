#!/bin/bash

PROFILE_DIR="$HOME/.chrome-remote-profile"

# 创建目录（如果不存在）
mkdir -p "$PROFILE_DIR"

# 删除锁文件（如果存在）
if [ -f "$PROFILE_DIR/SingletonLock" ]; then
    echo "Removing stale lock file..."
    rm -f "$PROFILE_DIR/SingletonLock"
fi

# 后台启动 Chrome，关闭输出
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="$PROFILE_DIR" \
  --no-first-run \
  --no-default-browser-check \
  > /dev/null 2>&1 &

echo "Chrome started in background with remote debugging on port 9222."
