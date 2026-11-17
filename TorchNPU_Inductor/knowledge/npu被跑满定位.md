npu被跑满

### 1. 查看哪些 NPU 跑满，以及占卡进程 PID

```
npu-smi info
```



重点看两部分：

- `AICore(%)`：哪张卡利用率高
- 下方 `Processes` 表：`Process id`、`Process name`、`Process memory`

想持续观察可以每 2 秒刷新一次，也只是查询：

```
watch -n 2 'date; npu-smi info'
```



按 `Ctrl+C` 退出 `watch`，只会退出监控界面，不会影响 NPU 任务。

执行以下只读命令：

```
PID=478114
echo '===== 进程及父进程 PID ====='ps -o user,pid,ppid,pgid,etime,time,%cpu,%mem,args -p "$PID"
echo '===== 父进程启动命令 ====='PPID1=$(ps -o ppid= -p "$PID" | tr -d ' ')ps -o user,pid,ppid,pgid,etime,args -p "$PPID1"
echo '===== 完整进程树 ====='pstree -aps "$PID"
echo '===== 容器/cgroup 信息 ====='sudo cat "/proc/$PID/cgroup"
echo '===== 进程工作目录 ====='sudo readlink -f "/proc/$PID/cwd"
```



通常**父进程启动命令**会直接暴露真正的入口，例如 `train.py`、`serve.py`、vLLM 或某个 benchmark。

也可以直接自动匹配 Docker 容器：

```
PID=478114

for c in $(docker ps -q); do
  if docker top "$c" -eo pid | awk 'NR > 1 {print $1}' | grep -qx "$PID"; then
    docker inspect -f '容器={{.Name}} 镜像={{.Config.Image}}' "$c"
    docker top "$c" -eo pid,ppid,user,etime,args
  fi
done
```



如果 Docker 需要权限，将循环里的 `docker` 都改为 `sudo docker`。

把 **`pstree`、父进程启动命令和 `cgroup`** 的输出发出来，基本就能确定它在跑什么、属于哪个容器。不需要 attach 进程，也不要用 `strace`/`gdb`，避免对任务产生影响。