# koutu

用于游戏 UI 抠图、Alpha Matte、Trimap 与前景重建实验。

当前流程：

1. 用矩形框指定目标 UI。
2. OpenCV GrabCut 生成初始主体 Mask。
3. 自动生成 Trimap：内部=前景，外部=背景，边缘 3~15px=Unknown。
4. PyMatting Closed-Form Matting 只求连续 Alpha。
5. `estimate_foreground_ml` 反推 Foreground RGB，避免直接拿截图 RGB 乘 Alpha。
6. 输出黑/白/红/蓝背景测试图，用于检查黑边、白边、背景染色、锯齿和阴影损失。

输出文件：

- `target_mask.png`
- `trimap.png`
- `alpha.png`
- `foreground.png`
- `final_rgba.png`
- `final_rgba_full.png`
- `test_black.png`
- `test_white.png`
- `test_red.png`
- `test_blue.png`

## GitHub Actions 运行

进入仓库 `Actions` → `UI Alpha Matting` → `Run workflow`。

参数：

- `image_path`：例如 `reference_ui.jpg`
- `bbox`：目标 UI 的 `x,y,w,h`
- `unknown_width`：建议 3~15，默认 8

运行完成后，在该次 Action 的 Artifacts 中下载 `matting-results`。

## 本地运行

```bash
pip install -r requirements.txt
python process.py --input reference_ui.jpg --bbox "x,y,w,h" --unknown-width 8 --output output
```

当前版本的“分割”是 CPU 友好的 GrabCut，适合作为第一版 Harness。后续可以把第一步替换成 SAM2 / RMBG / 自定义边缘与深度融合 Mask，而后面的 Trimap、Alpha Matte、Foreground Reconstruction 流程保持不变。
