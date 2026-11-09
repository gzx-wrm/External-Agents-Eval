# FeatureBench Lite 原始镜像持久缓存

独立下载工具。把固定Lite30的13个原始镜像保存到同一个OCI目录，共享压缩层只存一份。
不修改镜像内容、CUDA/Conda环境、评测提示词、模型、预算或评分；不改变正在运行的评测脚本。
OCI目录位于新100GiB盘，独立于原50GiB Docker数据目录，因此不受原评测的镜像清理影响。

采用官方`regctl v0.11.6`，二进制SHA256与官方发布记录核对一致，放在本次任务目录下。
用独立REGCTL_CONFIG和空Docker凭据配置，下载公开仓库，无模型API调用。
客户端支持[OCI目录复制与已有层复用](https://regclient.org/cli/regctl/image/copy/)。

流程：

1. 从原Lite公开清单取30题、13个唯一镜像；读取历史记录冻结的仓库、manifest摘要和config摘要。
2. 用regctl读取固定摘要的原始manifest，独立核对摘要，不依赖之后可能变化的latest。
3. 合并所有manifest、config与层的内容摘要，预算去重后空间；本次156个对象合计约81.55GiB。
4. 按镜像压缩大小顺序复制到同一OCI目录；下载请求并发上限3。镜像之间顺序执行，避免并发改写索引。
5. 逐镜像核对全部缓存对象的SHA256、大小和原始manifest摘要，之后才记为ready。
   共享对象只在内容/mtime未改变时复用已验证记录。
6. 复制失败保留已完成层并重试，最多5次；低于5GiB剩余空间时只终止自己的复制进程。
   任务结束写入result.json；进度写入monitor-status.json。

`cache_directory_bytes`包含正在下载的临时层，不代表已完成镜像。
`complete_blob_count`是大小匹配的对象数；`images_ready`须通过完整摘要验证。
文件锁阻止同一下载任务重复运行。中断后应先确认旧进程与复制子进程已退出，再运行同一计划继续。

大层加速器`range_copy.py`在原regctl复制前，通过官方端点的HTTPS Range定位下载地址，
用aria2逐层8路分段下载。已有顺序下载前缀经reflink和完整前缀哈希核对后继续使用。
临时文件完成全SHA256检查才移入原OCI摘要路径，再由regctl完成原manifest/index复制。
下载工具控制接口仅监听localhost，临时控制配置0600，进程结束后删除。
`stop-with-process`确保复制包装进程结束时下载子进程随之退出。

`range-progress.json`记录实际完成字节、连接数和速率；监控中的`effective_downloaded_bytes`
合并已完成对象与恢复中的内容。分段模式可能有稀疏暂存文件和保留的原临时文件，
目录逻辑大小不能用于估计完成比例。原始计划、启动记录和前缀校验保存在`acceleration/`。
完整旧数据仍可追溯；新运行未改变原镜像摘要或模型评测条件。

已验证：两个Docker格式的镜像能共用一个OCI层，两个标签都保留；manifest摘要和全部对象字节一致。
另用小样本验证了OCI导出→Docker26.1.1加载，加载后ImageID与原config摘要一致；测试镜像已删除。

离线加载示例（只在需要时执行）：

```sh
/root/featurebench-lite-image-cache-20261030-r1/bin/regctl image export \
  ocidir:///mnt/featurebench-cache/lite-oci:featurebench-specs_seaborn-instance_52738fbb \
  --name libercoders/featurebench-specs_seaborn-instance_52738fbb:latest \
  | docker -H unix:///run/featurebench-lite-docker.sock image load
```

Docker加载可保留ImageID；原仓库RepoDigest引用及现有`ensure_image`的缓存接入须另行适配验证，
本工具不自动切换当前评测的下载路径。不要将“OCI缓存可用”误报成“全部镜像已加载进Docker”。

测试：`python3 -m unittest discover -s tests -p test_featurebench_image_cache.py -v`。
