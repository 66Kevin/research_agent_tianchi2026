# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

# -*- coding: utf-8 -*-
import uuid
from urllib.parse import urlparse
import os
import fitz
import requests
from tqdm import tqdm
import subprocess
import re

from app.manus.gate.format_gate import format_check


@format_check()
def download_file(url, dest_path):
    chunk_size = 1024
    # 获取已下载文件大小（断点续传）
    resume_byte_pos = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    headers = {"Range": f"bytes={resume_byte_pos}-"}
    # 如果文件存在，直接覆盖（删除再下）
    if os.path.exists(dest_path):
        print(f"⚠️ 文件已存在，正在覆盖: {dest_path}")
        os.remove(dest_path)
        # 发起请求，获取文件大小
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('Content-Length', 0))

        with open(dest_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(dest_path)
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print(f"\n✅ 下载完成: {dest_path}")
    return f"\n✅ 下载完成: {dest_path}"


def convert_pdf_page_to_png(pdf_path: str, page_number: str) -> str:
    """
    将 PDF 文件中的指定页转换为 PNG 图片。

    Args:
        pdf_path (str): 输入的 PDF 文件路径。
        page_number (str): 需要转换的页码 (从 1 开始)。

    Returns:
        str: 描述转换结果的字符串，包含了输出文件的路径。
             如果发生错误，则返回错误信息。
    """
    output_dir = pdf_path
    # --- 输入验证 ---
    if not os.path.exists(pdf_path):
        return f"错误：PDF 文件不存在 -> {pdf_path}"

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            return f"错误：无法创建输出目录 -> {output_dir} | {e}"

    try:
        # 打开 PDF 文件
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"错误：无法打开或解析 PDF 文件 -> {pdf_path} | {e}"

    # --- 页码检查 ---
    if page_number <= 0 or page_number > doc.page_count:
        return f"错误：无效的页码。该 PDF 共有 {doc.page_count} 页，您指定的页码是 {page_number}。"

    # --- 核心转换逻辑 ---
    try:
        # PyMuPDF 的页码是从 0 开始的，所以需要减 1
        page = doc.load_page(page_number - 1)

        # 设置渲染的清晰度 (DPI)，值越高图片越清晰，文件也越大
        zoom = 2.0  # 缩放因子，2.0 对应 144 DPI (72 * 2)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # 构建输出文件名
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = f"{base_name}_page_{page_number}.png"
        output_path = os.path.join(output_dir, output_filename)

        # 保存为 PNG
        pix.save(output_path)

        return f"成功：已将 '{os.path.basename(pdf_path)}' 的第 {page_number} 页转换为 PNG 文件，并保存至 -> {output_path}"

    except Exception as e:
        return f"错误：在转换过程中发生未知错误 | {e}"

    finally:
        # 确保 PDF 文件被关闭
        if 'doc' in locals() and doc:
            doc.close()




def get_filename_from_url(url):
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)


def is_youtube_url(url):
    """
    判断是否为 YouTube 视频链接
    """
    parsed = urlparse(url)
    return "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc


def sanitize_filename(title):
    """
    简单清理非法字符，确保文件名合法
    """
    return re.sub(r'[\\/*?:"<>|]', "", title)


def get_video_title(url):
    """
    获取视频标题（用于构建文件名）
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-playlist", url],
            capture_output=True, text=True, check=True
        )
        title = result.stdout.strip()
        return sanitize_filename(title)
    except subprocess.CalledProcessError:
        return "default_name"


def download_youtube_audio_and_subs(url, lang="en", output_dir="."):
    """
    下载 YouTube 音频和自动字幕，并返回下载音频的绝对路径
    """
    if not is_youtube_url(url):
        print("⚠️ 该链接不是有效的 YouTube 视频链接，已跳过。")
        return None

    title = get_video_title(url)
    if not title:
        print("❌ 无法获取视频标题，可能是链接无效。")
        return None

    output_path = os.path.abspath(os.path.join(output_dir, f"{title}.mp4"))  # yt-dlp 默认音频格式常为 .webm 或 .m4a

    try:
        command = [
            "yt-dlp",
            "--write-auto-sub",
            "--no-playlist",
            "-o", output_path,
            url
        ]

        print(f"⏳ 正在下载：{title} ...")
        subprocess.run(command, check=True)
        print(f"✅ 下载完成！文件路径：{output_path}")
        compress_output_path = os.path.abspath(os.path.join(output_dir, f"{title}_compress.mp4"))
        compress_video_for_openai(output_path, compress_output_path)
        return compress_output_path

    except subprocess.CalledProcessError as e:
        print("❌ 下载失败，请检查 yt-dlp 是否正确安装，以及视频 URL 是否有效。")
        print("错误信息:", e)
        return None



def compress_video_for_openai(input_path, output_path=None, duration=30, width=640, fps=1, crf=24):
    """
    压缩视频并确保能够正常播放。
    :param input_path: 原始视频路径
    :param output_path: 输出路径（默认同目录添加 _compressed 后缀）
    :param duration: 保留时长（秒）
    :param width: 压缩后的视频宽度（高度自动等比）
    :param fps: 输出视频帧率
    :param crf: 压缩质量（越大越模糊，推荐 24~30）
    :return: 输出视频的绝对路径
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_compressed.mp4"

    output_path = os.path.abspath(output_path)

    command = [
        "ffmpeg",
        "-y",  # 覆盖已存在的输出文件
        "-i", input_path,
        "-t", str(duration),  # 保留前 N 秒
        "-vf", f"scale={width}:-1,fps={fps}",  # 压缩分辨率和帧率
        "-c:v", "libx264",  # 使用常见的视频编码器
        "-preset", "veryfast",  # 编码速度
        "-crf", str(crf),  # 控制视频质量（越大越模糊）
        "-c:a", "aac",  # 使用 AAC 音频编码器，兼容性更好
        "-b:a", "192k",  # 设置音频比特率
        "-movflags", "+faststart",  # 提前加载元数据，支持流式播放
        output_path
    ]

    try:
        print("🎬 正在压缩视频，请稍候...")
        subprocess.run(command, check=True)
        print(f"✅ 压缩完成！输出路径：{output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print("❌ 视频压缩失败，请检查 ffmpeg 是否正确安装。")
        print("错误信息:", e)
        return None
