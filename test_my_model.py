# test_my_model.py
import argparse
import os
import time
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

# 导入CompressAI必要的模块
import compressai
from compressai.models import CompressionModel
from compressai.zoo import image_models
from compressai.ops import compute_padding
import torch.nn.functional as F

def extract_epoch_from_checkpoint(checkpoint_path):
    """从checkpoint文件名中提取epoch信息，保留前导零（如果有）"""
    import re
    from pathlib import Path
    
    filename = Path(checkpoint_path).name
    
    # 匹配 checkpoint_epoch_020.pth.tar 或 checkpoint_epoch_200.pth.tar
    # 我们使用正则表达式来提取下划线后、点之前的数字部分
    match = re.search(r'epoch_(\d+)\.pth\.tar$', filename)
    if match:
        # 返回匹配的数字字符串，保留前导零
        return match.group(1)
    
    # 对于其他命名模式，比如 checkpoint_best_loss.pth.tar 或 checkpoint.pth.tar
    if 'best_loss' in filename:
        return "best"
    elif filename == 'checkpoint.pth.tar':
        return "final"
    
    # 如果都没有匹配到，返回文件名（不含扩展名）
    return Path(checkpoint_path).stem

def load_image(filepath: str) -> Image.Image:
    """加载图像"""
    return Image.open(filepath).convert("RGB")

def img2torch(img: Image.Image) -> torch.Tensor:
    """图像转Tensor"""
    return ToTensor()(img).unsqueeze(0)

def torch2img(x: torch.Tensor) -> Image.Image:
    """Tensor转图像"""
    return ToPILImage()(x.clamp_(0, 1).squeeze())

def pad(x, p=2**6):
    """填充图像到适合网络的大小"""
    h, w = x.size(2), x.size(3)
    pad, _ = compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode="constant", value=0)

def crop(x, size):
    """裁剪回原始大小"""
    H, W = x.size(2), x.size(3)
    h, w = size
    _, unpad = compute_padding(h, w, out_h=H, out_w=W)
    return F.pad(x, unpad, mode="constant", value=0)

def load_custom_model(model_arch, checkpoint_path, quality=3, metric="mse", device="cuda"):
    """加载自定义训练的模型"""
    # 1. 创建模型架构（与训练时相同）
    model_class = image_models[model_arch]
    net = model_class(quality=quality, metric=metric, pretrained=False).to(device)
    
    # 2. 加载训练好的权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查checkpoint结构并加载
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # 可能整个checkpoint就是state_dict
        state_dict = checkpoint
    
    # 处理DataParallel包装的key名（如果训练时用了多GPU）
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
# 修改后的 load_custom_model 函数尾部
    net.load_state_dict(state_dict)
    net.eval()
    
    # ===== 新增关键代码：更新熵模型 =====
    # 方法1：针对大多数CompressAI图像模型
    if hasattr(net, 'entropy_bottleneck'):
        net.entropy_bottleneck.update()
    # 方法2：更通用的方式，确保所有熵瓶颈层都更新
    for module in net.modules():
        if hasattr(module, 'entropy_bottleneck'):
            module.entropy_bottleneck.update()
        # 针对一些使用“gaussian_conditional”熵模型的架构（如bmshj2018-hyperprior）
        elif hasattr(module, 'gaussian_conditional'):
            module.gaussian_conditional.update()
    # ===== 新增代码结束 =====
    
    print(f"✓ 已加载模型: {model_arch}, 权重来自: {checkpoint_path}")
    print(f"  训练epoch: {checkpoint.get('epoch', '未知')}, "
          f"验证损失: {checkpoint.get('loss', '未知'):.4f}")
    return net

def test_model_on_image(model, image_path, output_dir="test_output", coder="ans", checkpoint_path=None):
    """用指定模型测试单张图像
    checkpoint_path: 模型权重文件路径，用于提取epoch信息命名文件
    """

    # 设置熵编码器
    compressai.set_entropy_coder(coder)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    img = load_image(image_path)
    x = img2torch(img).to(next(model.parameters()).device)
    h, w = x.size(2), x.size(3)
    
    print(f"测试图像: {image_path} ({w}x{h})")
    
    # 编码（压缩）
    start_time = time.time()
    x_padded = pad(x)
    
    with torch.no_grad():
        out = model.compress(x_padded)
    
    enc_time = time.time() - start_time
    
    # 计算压缩率
    strings = out["strings"]
    shape = out["shape"]
    
    # 估算压缩后大小（字节）
    compressed_size = sum(len(s[0]) for s in strings)
    original_size = os.path.getsize(image_path)
    num_pixels = h * w
    
    bpp = compressed_size * 8.0 / num_pixels  # 每像素比特数
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    print(f"编码时间: {enc_time:.2f}s")
    print(f"原始大小: {original_size:,} 字节")
    print(f"压缩后大小: ~{compressed_size:,} 字节")
    print(f"压缩率: {compression_ratio:.2f}:1")
    print(f"比特率: {bpp:.3f} bpp")
    
    # 保存压缩数据（可选）
    compressed_filename = Path(image_path).stem + "_compressed.bin"
    compressed_path = os.path.join(output_dir, compressed_filename)
    
    # 解码（重建）
    start_time = time.time()
    with torch.no_grad():
        out_dec = model.decompress(strings, shape)
    
    dec_time = time.time() - start_time
    
    x_hat = crop(out_dec["x_hat"], (h, w))
    
    # 保存重建图像（使用新的命名规则）
    reconstructed_filename = f"epoch_{extract_epoch_from_checkpoint(checkpoint_path)}.png"
    reconstructed_path = os.path.join(output_dir, reconstructed_filename)
    
    rec_img = torch2img(x_hat)
    rec_img.save(reconstructed_path)
    
    print(f"解码时间: {dec_time:.2f}s")
    print(f"重建图像已保存: {reconstructed_path}")
    
    return {
        "bpp": bpp,
        "compression_ratio": compression_ratio,
        "enc_time": enc_time,
        "dec_time": dec_time,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }

def main():
    parser = argparse.ArgumentParser(description="测试自定义训练的CompressAI模型")
    parser.add_argument(
        "--model-arch", 
        type=str, 
        default="mbt2018-mean",  # 根据你实际训练的模型修改
        help="模型架构（必须与训练时相同）"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="模型权重文件路径（.pth.tar）"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="测试图像路径"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="test_output",
        help="输出目录"
    )
    parser.add_argument(
        "--quality", 
        type=int, 
        default=3,
        help="质量等级（必须与训练时相同）"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="mse",
        choices=["mse", "ms-ssim"],
        help="度量标准（必须与训练时相同）"
    )
    parser.add_argument("--cuda", action="store_true", help="使用GPU")
    
    args = parser.parse_args()
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 1. 加载自定义模型
    model = load_custom_model(
        model_arch=args.model_arch,
        checkpoint_path=args.checkpoint,
        quality=args.quality,
        metric=args.metric,
        device=device
    )
    
     # 2. 测试模型（现在传递了checkpoint_path参数）
    results = test_model_on_image(
        model=model,
        image_path=args.image,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint  # 新增参数
    )
    
    print("\n" + "="*50)
    print(f"测试完成！")
    print(f"模型: {args.model_arch}, 权重: {Path(args.checkpoint).name}")
    print(f"比特率: {results['bpp']:.3f} bpp")
    print(f"压缩比: {results['compression_ratio']:.2f}:1")
    print("="*50)

if __name__ == "__main__":
    main()
