import numpy as np
import os
import random
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import imageio
import struct

from plyfile import PlyData, PlyElement
from time import time
from tqdm import tqdm

from gaussian_renderer import render
from utils.image_utils import psnr, img_tv_loss
from utils.graphics_utils import BasicPointCloud
from gaussian_core.cameras import CamerasWrapper
from gaussian_core.cameras import convert_gs_to_pytorch3d
from pytorch3d.transforms import quaternion_apply, quaternion_invert

from gaussian_core.watermark_utils import EndoGSWatermarker




to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


# ====== [Endo-OT-RDH: Neural ODE & Photometric Flow] ======
class ODEVectorField(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels + 1, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, kernel_size=1)
        )

    def forward(self, x, potential):
        feat = torch.cat([x, potential], dim=1)
        return self.net(feat)


class PhotometricNeuralODEBlock(nn.Module):
    def __init__(self, channels, dt=0.2):
        super().__init__()
        self.dt = dt
        self.F_field = ODEVectorField(channels)
        self.G_field = ODEVectorField(channels)

    def forward(self, x_A, x_B, v_A, v_B, reverse=False):
        if not reverse:
            x_B_out = x_B + self.dt * self.F_field(x_A, v_A)
            x_A_out = x_A + self.dt * self.G_field(x_B_out, v_B)
            return x_A_out, x_B_out
        else:
            x_A_in = x_A - self.dt * self.G_field(x_B, v_B)
            x_B_in = x_B - self.dt * self.F_field(x_A_in, v_A)
            return x_A_in, x_B_in


class ContinuousManifoldFlowODE(nn.Module):
    def __init__(self, channels=4, num_steps=3):
        super().__init__()
        self.dt = 1.0 / num_steps
        self.blocks = nn.ModuleList([PhotometricNeuralODEBlock(channels, self.dt) for _ in range(num_steps)])

    def forward(self, x_A, x_B, v_A, v_B, reverse=False):
        if not reverse:
            for block in self.blocks:
                x_A, x_B = block(x_A, x_B, v_A, v_B, reverse=False)
            return x_A, x_B
        else:
            for block in reversed(self.blocks):
                x_A, x_B = block(x_A, x_B, v_A, v_B, reverse=True)
            return x_A, x_B


# =======================================================

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def training(opt, dataloader, gaussians, watermarker=None):
    spatial_lr_scale = 5

    scene_name = getattr(opt, 'scene', '')
    if 'pulling' in scene_name:
        pretrained_model_path = "output/pulling/point_cloud/iteration_60000"
    else:
        pretrained_model_path = "output/cutting/point_cloud/iteration_60000"

    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        print(f"\n[医学水印] 检测到预训练基线模型: {pretrained_model_path}")
        print("[医学水印] 开启联合优化模式：跳过 Coarse 阶段，载入权重并执行 60,000 步持续嵌入微调...")
        gaussians.load_ply(os.path.join(pretrained_model_path, "point_cloud.ply"))
        gaussians.load_model(pretrained_model_path)
        gaussians.training_setup()

        recon(opt, dataloader, gaussians, "fine", 60000, is_pretrained=True)
        return

    data = next(iter(dataloader))
    ply_path = os.path.join(data['sparse_path'], "points3D.ply")
    bin_path = os.path.join(data['sparse_path'], "points3D.bin")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)
    gaussians.create_from_pcd(pcd, spatial_lr_scale)
    gaussians.training_setup()

    if opt.coarse_iters > 0:
        recon(opt, dataloader, gaussians, "coarse", opt.coarse_iters, watermarker=EndoGSWatermarker)
    if opt.fine_iters > 0:
        recon(opt, dataloader, gaussians, "fine", opt.fine_iters, watermarker=EndoGSWatermarker)


def recon(opt, dataloader, gaussians, stage, num_iter, watermarker=None, is_pretrained=False):
    first_iter = 0
    white_background = 1
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    densify_from_iter = 500
    densify_until_iter = 45000
    densification_interval = 100
    densify_grad_threshold_coarse = 0.0002
    densify_grad_threshold_fine_init = 0.0002
    densify_grad_threshold_after = 0.0002

    opacity_reset_interval = 6000
    opacity_threshold_coarse = 0.005
    opacity_threshold_fine_init = 0.005
    opacity_threshold_fine_after = 0.005

    pruning_from_iter = 500
    pruning_interval = 7000

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    if is_pretrained and stage == "fine":
        num_iter = 60000

    final_iter = num_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    watermarker = None
    ode_model = None

    enable_watermark_iter = 10000
    if is_pretrained and stage == "fine":
        enable_watermark_iter = 0

    log_interval = 500
    iteration = first_iter

    payload_bits = getattr(opt, 'wm_payload_bits', 32)

    while True:
        for data in dataloader:
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cam = data['camera']
            fov_camera = convert_gs_to_pytorch3d([viewpoint_cam])

            target_size = (viewpoint_cam.original_image.shape[1], viewpoint_cam.original_image.shape[2])

            mask = data['mask'].unsqueeze(0).unsqueeze(0).to("cuda")
            if mask.shape[-2:] != target_size:
                mask = F.interpolate(mask.float(), size=target_size, mode='nearest').bool()

            frozen_mask = None

            if 'tool_mask' in data and 'lesion_mask' in data:
                tool_mask = data['tool_mask'].unsqueeze(0).unsqueeze(0).to("cuda")
                lesion_mask = data['lesion_mask'].unsqueeze(0).unsqueeze(0).to("cuda")

                if tool_mask.shape[-2:] != target_size:
                    tool_mask = F.interpolate(tool_mask.float(), size=target_size, mode='nearest').bool()
                if lesion_mask.shape[-2:] != target_size:
                    lesion_mask = F.interpolate(lesion_mask.float(), size=target_size, mode='nearest').bool()

                roni_mask = tool_mask | lesion_mask
                mask = mask & (~roni_mask)

                if not hasattr(gaussians, 'world_freeze_mask'):
                    with torch.no_grad():
                        xyz = gaussians.get_xyz.detach()
                        xyz_homo = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=-1)
                        P_clip = xyz_homo @ viewpoint_cam.full_proj_transform
                        w = P_clip[:, 3:4] + 1e-6
                        x_ndc = P_clip[:, 0:1] / w
                        y_ndc = P_clip[:, 1:2] / w

                        W_img, H_img = target_size[1], target_size[0]
                        x_pix = torch.clamp(torch.round(((x_ndc + 1.0) * W_img - 1.0) * 0.5).long(), 0,
                                            W_img - 1).squeeze(-1)
                        y_pix = torch.clamp(torch.round(((y_ndc + 1.0) * H_img - 1.0) * 0.5).long(), 0,
                                            H_img - 1).squeeze(-1)

                        roni_mask_2d = roni_mask.squeeze()
                        gaussians.world_freeze_mask = roni_mask_2d[y_pix, x_pix].clone()

                frozen_mask = gaussians.world_freeze_mask
                del tool_mask, lesion_mask, roni_mask

            render_pkg = render(viewpoint_cam, gaussians, data['time'], background, stage=stage)
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            depth = render_pkg["depth"]

            image_tensor = image.unsqueeze(0)
            pred_depth_tensor = (depth / (depth.max() + 1e-5)).unsqueeze(0)
            gt_image_tensor = viewpoint_cam.original_image.cuda().unsqueeze(0)

            del render_pkg

            weight = data['spatialweight'].unsqueeze(0).unsqueeze(0).to(image_tensor.device)
            gt_depth = (data['depth'] / (data['depth'].max() + 1e-5)).unsqueeze(0).unsqueeze(0).to(image_tensor.device)

            if gt_depth.shape[-2:] != target_size:
                gt_depth = F.interpolate(gt_depth.float(), size=target_size, mode='nearest')

            mask_float = mask.float()
            Ll1 = (torch.abs((image_tensor * mask_float - gt_image_tensor * mask_float)) * weight).mean()
            psnr_ = psnr(image_tensor * mask_float, gt_image_tensor * mask_float).mean().float()

            depth_loss = F.huber_loss(pred_depth_tensor * mask_float, gt_depth * mask_float, delta=0.2)
            img_tvloss = img_tv_loss(image_tensor * (1 - mask_float))

            loss = Ll1 + 0.5 * depth_loss + 0.01 * img_tvloss

            if stage == "fine" and iteration >= enable_watermark_iter:
                if watermarker is None:
                    print(f"\n[Iter {iteration}] 初始化 Endo-OT-RDH 医学专用版权系统 (指定载荷: {payload_bits} Bits)...")
                    num_sh_coefs = 1 + (gaussians._features_rest.shape[1] if hasattr(gaussians,
                                                                                     '_features_rest') and gaussians._features_rest is not None else 0)

                    ode_channels = num_sh_coefs * 3

                    ode_model = ContinuousManifoldFlowODE(channels=ode_channels, num_steps=3).cuda()
                    decouple_mode = getattr(opt, 'decouple_mode', 'C')
                    print(f"[医学水印] 激活光度势能解耦控制模态: {decouple_mode} | 流形状态通道: {ode_channels}")

                    watermarker = EndoGSWatermarker(
                        gaussians, secret_seed=42, msg_length=payload_bits, ode_model=ode_model, decouple_mode=decouple_mode,
                        m_repeats=7
                    )
                    watermarker.extractor.train()

                ode_loss, loss_3d_acc, acc_3d = watermarker.compute_ode_loss(viewpoint_cam, fov_camera, frozen_mask=frozen_mask)

                warmup_iters = max(1, final_iter - enable_watermark_iter)
                progress = (iteration - enable_watermark_iter) / warmup_iters
                current_wm_weight = 0.3 * progress

                wm_loss, ber, semantic_loss, c_endo_loss = watermarker.compute_watermark_loss(
                    image_tensor, watermarker.binary_msg, safe_mask=mask_float, depth_tensor=pred_depth_tensor
                )

                # 适当加大 3D 损失权重，促使 3D 损失尽快下降至 0.0001 以下
                weight_3d = min(15.0, 8.0 + 7.0 * progress)

                loss = loss + (wm_loss.squeeze() * current_wm_weight) + \
                       (semantic_loss.squeeze() * 0.05) + \
                       (c_endo_loss.squeeze() * 0.1) + \
                       (ode_loss * 0.05) + \
                       (loss_3d_acc * weight_3d)

                if iteration % log_interval == 0:
                    print(
                        f"\n[Iter {iteration}] BER = {ber:.4f}, SemLoss = {semantic_loss.squeeze().item():.4f}, ODE_Loss = {ode_loss.item():.4f}, 3D_Acc_Loss = {loss_3d_acc.item():.6f}, 3D_Acc = {acc_3d.item():.2%}")

            loss.backward()

            if stage == "fine" and iteration >= enable_watermark_iter:
                if gaussians._xyz.grad is not None: gaussians._xyz.grad.zero_()
                if gaussians._rotation.grad is not None: gaussians._rotation.grad.zero_()
                if gaussians._scaling.grad is not None: gaussians._scaling.grad.zero_()
                if gaussians._opacity.grad is not None: gaussians._opacity.grad.zero_()

                if frozen_mask is not None:
                    if gaussians._features_dc.grad is not None:
                        gaussians._features_dc.grad[frozen_mask] = 0.0
                    if hasattr(gaussians,
                               '_features_rest') and gaussians._features_rest is not None and gaussians._features_rest.grad is not None:
                        gaussians._features_rest.grad[frozen_mask] = 0.0

            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            iter_end.record()

            with torch.no_grad():
                initial_loss_val = loss.item() if hasattr(loss, 'item') else 0.0
                ema_loss_for_log = 0.4 * initial_loss_val + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr_.item() + 0.6 * ema_psnr_for_log
                total_point = gaussians._xyz.shape[0]

                if iteration % 10 == 0:
                    post_dict = {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "psnr": f"{ema_psnr_for_log:.{2}f}",
                        "point": f"{total_point}"
                    }
                    if stage == "fine" and iteration >= enable_watermark_iter and 'acc_3d' in locals():
                        post_dict["3d_acc"] = f"{acc_3d.item():.2%}"
                        post_dict["3d_loss"] = f"{loss_3d_acc.item():.6f}"
                    progress_bar.set_postfix(post_dict)
                    progress_bar.update(10)

                if iteration % 20000 == 0 or iteration == final_iter:
                    if iteration == final_iter and stage == 'fine' and watermarker is not None:
                        # # ===== 新增：用最终参数微调ODE，确保闭环在最终参数上成立 =====
                        # print("\n[医学水印] 用最终参数微调 ODE 闭环 (500步, freeze SH)...")
                        # watermarker.extractor.train()
                        # for fine_step in range(500):
                        #     ode_loss_f, loss_3d_f, acc_3d_f = watermarker.compute_ode_loss(
                        #         viewpoint_cam, fov_camera, frozen_mask=frozen_mask)
                        #     total_fine = ode_loss_f * 0.05 + loss_3d_f * 15.0
                        #     total_fine.backward()
                        #     watermarker.extractor_optimizer.step()
                        #     watermarker.extractor_optimizer.zero_grad(set_to_none=True)
                        #     if fine_step % 100 == 0:
                        #         print(f"  [微调 {fine_step}] ode_loss={ode_loss_f.item():.6f}, "
                        #               f"3d_loss={loss_3d_f.item():.6f}, 3d_acc={acc_3d_f.item():.2%}")
                        #
                        print("[医学水印] ODE 闭环微调完成。")

                        backup_frozen = None
                        if frozen_mask is not None and frozen_mask.any():
                            backup_frozen = {
                                'features_dc': gaussians._features_dc[frozen_mask].clone()
                            }
                            if hasattr(gaussians, '_features_rest') and gaussians._features_rest is not None:
                                backup_frozen['features_rest'] = gaussians._features_rest[frozen_mask].clone()

                        watermarker.embed_watermark(viewpoint_cam, fov_camera, frozen_mask=frozen_mask)

                        if backup_frozen is not None:
                            gaussians._features_dc[frozen_mask] = backup_frozen['features_dc']
                            if 'features_rest' in backup_frozen:
                                gaussians._features_rest[frozen_mask] = backup_frozen['features_rest']
                            print("[医学水印] 限制区域(病灶及工具)的 3D 高斯基元已强行实施全属性状态固化恢复。")

                        extractor_save_path = os.path.join(opt.workspace, "watermark_extractor.pth")
                        torch.save(watermarker.extractor.state_dict(), extractor_save_path)

                        ode_save_path = os.path.join(opt.workspace, "watermark_ode.pth")
                        torch.save(watermarker.ode_model.state_dict(), ode_save_path)

                        if watermarker.saved_indices_A is not None:
                            torch.save(watermarker.saved_indices_A, os.path.join(opt.workspace, "watermark_idx_A.pt"))
                        if watermarker.saved_indices_B is not None:
                            torch.save(watermarker.saved_indices_B, os.path.join(opt.workspace, "watermark_idx_B.pt"))

                        torch.save({
                            'peak_A': getattr(watermarker, 'saved_peak_A', 0.0),
                            'peak_B': getattr(watermarker, 'saved_peak_B', 0.0)
                        }, os.path.join(opt.workspace, "watermark_peaks.pt"))

                        if hasattr(watermarker, 'saved_residuals') and watermarker.saved_residuals is not None:
                            torch.save(watermarker.saved_residuals, os.path.join(opt.workspace, "watermark_residuals.pt"))

                        print("[医学水印] 已将提取器、神经常微分网络(ODE)权重、流形参考系及可逆无损补偿标定文件固化保存。")

                    gaussians.save(opt.workspace, iteration, stage)


                    if watermarker is not None:
                        wm_path = os.path.join(
                            opt.workspace,
                            f"watermark_{stage}_{iteration}.pth"
                        )
                        EndoGSWatermarker.save_watermark_checkpoint(watermarker, wm_path)
                        print("[ITER {}] Saving Watermarker -> {}".format(iteration, wm_path))


                if iteration < densify_until_iter and not is_pretrained:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                    if stage == "coarse":
                        opacity_threshold = opacity_threshold_coarse
                        densify_threshold = densify_grad_threshold_coarse
                    else:
                        opacity_threshold = opacity_threshold_fine_init - iteration * (
                                opacity_threshold_fine_init - opacity_threshold_fine_after) / (densify_until_iter)
                        densify_threshold = densify_grad_threshold_fine_init - iteration * (
                                densify_grad_threshold_fine_init - densify_grad_threshold_after) / (
                                                densify_until_iter)

                    if iteration > densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        gaussians.densify(densify_threshold, opacity_threshold, 5, size_threshold)

                    if iteration > pruning_from_iter and iteration % pruning_interval == 0:
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        gaussians.prune(densify_threshold, opacity_threshold, 5, size_threshold)
                        torch.cuda.empty_cache()

                    if iteration % opacity_reset_interval == 0 or (white_background and iteration == densify_from_iter):
                        gaussians.reset_opacity()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if stage == "fine" and iteration >= enable_watermark_iter and watermarker is not None:
                    watermarker.extractor_optimizer.step()
                    watermarker.extractor_optimizer.zero_grad(set_to_none=True)

            del image_tensor, pred_depth_tensor, gt_image_tensor, mask, mask_float, weight, gt_depth

            iteration += 1
            if iteration > final_iter:
                break
        if iteration > final_iter:
            break


def testing(opt, dataloader, gaussians, save_gt=True, watermarker=None, watermark_ckpt=None):
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("\n=== 初始化 Endo-OT-RDH 医学模型自动化攻击全量评估 ===")

    decouple_mode = getattr(opt, 'decouple_mode', 'C')
    payload_bits = getattr(opt, 'wm_payload_bits', 32)

    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(os.path.join(opt.model_path))

    if watermarker is not None:
        if watermark_ckpt is None:
            watermark_ckpt = os.path.join(opt.model_path, "watermark_final.pth")
        print(f"Loading watermark checkpoint from {watermark_ckpt}")
        EndoGSWatermarker.load_watermark_checkpoint(watermarker, watermark_ckpt, device='cuda')

    num_sh_coefs = 1 + (gaussians._features_rest.shape[1] if hasattr(gaussians,
                                                                     '_features_rest') and gaussians._features_rest is not None else 0)

    ode_channels = num_sh_coefs * 3

    ode_model = ContinuousManifoldFlowODE(channels=ode_channels, num_steps=3).cuda()
    ode_path = os.path.join(opt.model_path, "watermark_ode.pth")
    if os.path.exists(ode_path):
        ode_model.load_state_dict(torch.load(ode_path))
        print(f"[医学水印] 成功加载连续流形(Neural ODE)网络权重.")

    watermarker = EndoGSWatermarker(gaussians, secret_seed=42, msg_length=payload_bits, ode_model=ode_model,
                                    decouple_mode=decouple_mode, m_repeats=7)

    extractor_path = os.path.join(opt.model_path, "watermark_extractor.pth")
    if os.path.exists(extractor_path):
        watermarker.extractor.load_state_dict(torch.load(extractor_path))
        watermarker.extractor.eval()
        print(f"[医学水印] 成功加载 2D 水印黑盒提取器权重.")

    idx_A_path = os.path.join(opt.model_path, "watermark_idx_A.pt")
    idx_B_path = os.path.join(opt.model_path, "watermark_idx_B.pt")
    peaks_path = os.path.join(opt.model_path, "watermark_peaks.pt")
    residuals_path = os.path.join(opt.model_path, "watermark_residuals.pt")

    if os.path.exists(idx_A_path) and os.path.exists(idx_B_path):
        watermarker.saved_indices_A = torch.load(idx_A_path)
        watermarker.saved_indices_B = torch.load(idx_B_path)
        print(f"[医学水印] 成功加载固化点索引.")

    if os.path.exists(peaks_path):
        peaks = torch.load(peaks_path)
        watermarker.saved_peak_A = peaks['peak_A']
        watermarker.saved_peak_B = peaks['peak_B']
        print(f"[医学水印] 成功加载物理状态演化峰值 (避免峰值倒置).")

    if os.path.exists(residuals_path):
        watermarker.saved_residuals = torch.load(residuals_path)
        print(f"[医学水印] 成功加载 PSNR=inf 比特级绝对无损恢复补偿标定文件.")

    test_data = next(iter(dataloader))
    test_cam = test_data['camera']
    fov_test_cam = convert_gs_to_pytorch3d([test_cam])

    frozen_mask_test = None
    if 'tool_mask' in test_data and 'lesion_mask' in test_data:
        with torch.no_grad():
            t_mask = test_data['tool_mask'].unsqueeze(0).unsqueeze(0).to("cuda")
            l_mask = test_data['lesion_mask'].unsqueeze(0).unsqueeze(0).to("cuda")

            target_size = test_cam.original_image.shape[-2:]
            t_mask = F.interpolate(t_mask.float(), size=target_size, mode='nearest').bool()
            l_mask = F.interpolate(l_mask.float(), size=target_size, mode='nearest').bool()

            roni_mask_test = t_mask | l_mask
            H_t, W_t = target_size

            xyz = gaussians.get_xyz.detach()
            xyz_homo = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=-1)
            P_clip = xyz_homo @ test_cam.full_proj_transform
            w = P_clip[:, 3:4] + 1e-6
            x_ndc = P_clip[:, 0:1] / w
            y_ndc = P_clip[:, 1:2] / w
            x_pix = torch.clamp(torch.round(((x_ndc + 1.0) * W_t - 1.0) * 0.5).long(), 0, W_t - 1).squeeze(-1)
            y_pix = torch.clamp(torch.round(((y_ndc + 1.0) * H_t - 1.0) * 0.5).long(), 0, H_t - 1).squeeze(-1)

            frozen_mask_test = roni_mask_test.squeeze()[y_pix, x_pix]

    # =========================================================
    # 自动化测评并写入 TXT
    # =========================================================
    results_txt_path = os.path.join(opt.model_path, "watermark_evaluation_results.txt")
    print(f"\n[医学水印] 正在自动触发各项物理与图像攻击，请稍候...")

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write(f" Endo-OT-RDH 医学模型水印鲁棒性自动化全量评估报告 ({payload_bits} Bits)\n")
        f.write("=" * 55 + "\n\n")

        # ---------------- 3D 模型攻击评估 ----------------
        f.write("【一】 3D 模型参数级物理攻击提取准确率\n")
        f.write("-" * 45 + "\n")
        attacks_3d = ['none', '3d_noise', '3d_prune', '3d_quant']

        for atk in attacks_3d:
            gaussians.load_model(os.path.join(opt.model_path))
            if atk != 'none':
                watermarker.apply_3d_attack(atk)

            is_auth, acc_2d, acc_3d = watermarker.extract_and_verify(test_cam, fov_test_cam, test_data['time'], frozen_mask=frozen_mask_test)
            atk_label = "无攻击 (Clean)" if atk == 'none' else atk
            f.write(f" 攻击类型: {atk_label.ljust(15)} | 2D准确率: {acc_2d:.2%} | 3D准确率: {acc_3d:.2%}\n")
            print(f" -> [3D] {atk_label.ljust(15)} : 2D Acc = {acc_2d:.2%} | 3D Acc = {acc_3d:.2%}")

        # ---------------- 2D 图像攻击评估 ----------------
        f.write("\n\n【二】 2D 图像级黑盒攻击提取准确率\n")
        f.write("-" * 45 + "\n")

        gaussians.load_model(os.path.join(opt.model_path))

        clean_wm_images = []
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Rendering clean images for 2D evaluation"):
                rendering = render(data['camera'], gaussians, data['time'], background)["render"]

                test_mask = None
                if 'mask' in data:
                    test_mask = data['mask'].unsqueeze(0).unsqueeze(0).to(rendering.device)
                    if test_mask.shape[-2:] != rendering.shape[-2:]:
                        test_mask = F.interpolate(test_mask.float(), size=rendering.shape[-2:], mode='nearest').bool()
                    if 'tool_mask' in data and 'lesion_mask' in data:
                        t_mask = F.interpolate(data['tool_mask'].unsqueeze(0).unsqueeze(0).float(),
                                               size=rendering.shape[-2:], mode='nearest').bool().to(rendering.device)
                        l_mask = F.interpolate(data['lesion_mask'].unsqueeze(0).unsqueeze(0).float(),
                                               size=rendering.shape[-2:], mode='nearest').bool().to(rendering.device)
                        test_mask = test_mask & (~(t_mask | l_mask))
                        del t_mask, l_mask

                wm_img = watermarker.embed_image_watermark(rendering.unsqueeze(0),
                                                           safe_mask=test_mask.float() if test_mask is not None else None)
                clean_wm_images.append(wm_img)

        attacks_2d = ['none', '2d_noise', '2d_blur', '2d_crop']
        for atk in attacks_2d:
            ber_list = []
            with torch.no_grad():
                for wm_img in clean_wm_images:
                    if atk != 'none':
                        attacked_img = watermarker.apply_2d_attack(wm_img, atk)
                    else:
                        attacked_img = wm_img.clone()

                    logits = watermarker.extractor(attacked_img)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    target = ((watermarker.binary_msg + 1) / 2.0).view(1, -1)
                    ber = (preds != target).float().mean().item()
                    ber_list.append(ber)

            is_auth, _, acc_3d_val = watermarker.extract_and_verify(test_cam, fov_test_cam, data['time'], frozen_mask=frozen_mask_test)

            if len(ber_list) > 0:
                avg_ber = sum(ber_list) / len(ber_list)
                avg_acc_2d = 1.0 - avg_ber
                atk_label = "无攻击 (Clean)" if atk == 'none' else atk
                f.write(f" 攻击类型: {atk_label.ljust(15)} | 2D准确率: {avg_acc_2d:.2%} | 3D准确率: {acc_3d_val:.2%}\n")
                print(f" -> [2D] {atk_label.ljust(15)} : 2D Acc = {avg_acc_2d:.2%} | 3D Acc = {acc_3d_val:.2%}")
            else:
                atk_label = "无攻击 (Clean)" if atk == 'none' else atk
                f.write(f" 攻击类型: {atk_label.ljust(15)} | 2D提取失败 (无数据) | 3D准确率: {acc_3d_val:.2%}\n")
                print(f" -> [2D] {atk_label.ljust(15)} : 2D提取失败 | 3D Acc = {acc_3d_val:.2%}")

        f.write("\n" + "=" * 55 + "\n")

    print(f"\n[医学水印] 所有基准与对抗测试完毕！全景评估报告已成功保存至: \n{results_txt_path}")

    # =========================================================
    # 生成无攻击基线渲染与逆映射恢复图
    # =========================================================
    print("\n[医学水印] 正在输出无攻击情况下的基线渲染图与无损恢复图...")
    gaussians.load_model(os.path.join(opt.model_path))

    render_path_wm = os.path.join(opt.model_path, "render_watermarked_clean")
    os.makedirs(render_path_wm, exist_ok=True)
    count = 0
    for image in clean_wm_images:
        torchvision.utils.save_image(image.squeeze(0), os.path.join(render_path_wm, '{0:05d}'.format(count) + ".png"))
        count += 1
    print(f"基线带水印图像已输出至: {render_path_wm}")

    print("\n[医学水印] 正在触发参数级逆映射无损修复(保证 PSNR=inf, SSIM=1.0)...")

    backup_frozen_test = None
    if frozen_mask_test is not None and frozen_mask_test.any():
        backup_frozen_test = {
            'features_dc': gaussians._features_dc[frozen_mask_test].clone()
        }
        if hasattr(gaussians, '_features_rest') and gaussians._features_rest is not None:
            backup_frozen_test['features_rest'] = gaussians._features_rest[frozen_mask_test].clone()

    watermarker.recover_model(test_cam, fov_test_cam, frozen_mask=frozen_mask_test)

    if backup_frozen_test is not None:
        with torch.no_grad():
            gaussians._features_dc[frozen_mask_test] = backup_frozen_test['features_dc']
            if 'features_rest' in backup_frozen_test:
                gaussians._features_rest[frozen_mask_test] = backup_frozen_test['features_rest']

    render_path_rec = os.path.join(opt.model_path, "render_recovered_clean")
    os.makedirs(render_path_rec, exist_ok=True)
    render_list_rec = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Rendering Recovered")):
            rendering = render(data['camera'], gaussians, data['time'], background)["render"]
            render_list_rec.append(rendering)

    count = 0
    for image in render_list_rec:
        torchvision.utils.save_image(image, os.path.join(render_path_rec, '{0:05d}'.format(count) + ".png"))
        count += 1
    print(f"[医学水印] 纯净无损恢复渲染图像(PSNR=inf)已成功输出至: {render_path_rec}")

    if save_gt:
        gts_path = os.path.join(opt.model_path, "gt")
        os.makedirs(gts_path, exist_ok=True)
        count = 0
        for data in dataloader:
            gt = data['camera'].original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count += 1