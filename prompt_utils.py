import torch
from torch import Tensor
from PIL import Image
import base64
import requests
from io import BytesIO
from conversation import conv_templates
from blip_process import BlipImageEvalProcessor
import glob
from natsort import natsorted
from typing import Union
from typing import List
from pdf_file_utils import pdf_to_image


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

device = "cuda"
dtype = torch.bfloat16
stop_token_id = 151645
image_token_len = 256
image_processor = BlipImageEvalProcessor(image_size=1024)


def get_stop_token_id():
    return stop_token_id


def __get_prompt_input(image_size: int = 1, type: str = 'plain', multi_page: bool = False):
    image_token_len = 256
    qs = ''
    if type.lower() == 'plain':
        qs += 'OCR: '
    else:
        qs += "OCR with format: "

    if multi_page:
        qs = 'OCR with format across multi pages: '

    qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_PATCH_TOKEN * image_token_len * image_size }{DEFAULT_IM_END_TOKEN}\n{qs}"
    # 配置对话模板
    conv_mode = "mpt"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def __load_image_tensor(image_file):
    # 获取单个图片的tensor
    if isinstance(image_file, Image.Image):
        image = image_file
    elif isinstance(image_file, bytes):
        # 从bytes加载图像并转换为RGB格式
        image = Image.open(BytesIO(image_file)).convert('RGB')
    elif isinstance(image_file, str):
        # 从文件路径或url加载图像并转换为RGB格式
        if image_file.startswith('http'):
            # 重试逻辑
            try_count = 0
            while try_count < 3:
                try:
                    response = requests.get(image_file, timeout=5)
                    image = Image.open(
                        BytesIO(response.content)).convert('RGB')
                    break
                except Exception as e:
                    print(f"Error loading image from {image_file}: {e}")
                    try_count += 1
        elif image_file.startswith("data:image/"):
            # base64编码的 data:image/png;base64,
            image_data = base64.b64decode(image_file.split('base64,')[1])
            image_bytes = BytesIO(image_data)
            image = Image.open(image_bytes).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')

    image_tensor = image_processor(image).to(
        device=device, dtype=torch.bfloat16)
    image.close()
    return image_tensor


def __load_image_tensor_list(image_file_list):
    # 获取多个图片的tensors
    image_tensors = []
    for image_file in image_file_list:
        image_tensor = __load_image_tensor(image_file)
        image_tensors.append(image_tensor)
    return image_tensors


def get_got_prompts(image_list: Union[str, List[str], List[List[str]]], type: str = None, box: List[List[int]] = None):
    if image_list is None:
        raise ValueError("image_list must be a string or a list of strings")
    result_prompt_inputs = []
    result_prompt_images = []
    multi_page = False

    # 增补支持单个pdf文档
    if isinstance(image_list, str) and image_list.endswith(".pdf"):
        image_list = pdf_to_image(image_list)

    image_ext_names = ['.png', '.jpg', '.jpeg', '.webp']

    if isinstance(image_list, str):
        if image_list.startswith("data:image/") or image_list.startswith("http") or "."+image_list.split('.')[-1].lower() in image_ext_names:
            if box is None:
                image = __load_image_tensor(image_list)
                input = __get_prompt_input(
                    image_size=1, type=type, multi_page=multi_page)
                result_prompt_inputs.append(input)
                result_prompt_images.append(image)
            elif isinstance(box[0], int):
                orginal_image = Image.open(image_list)
                x1, y1, x2, y2 = box
                image = __load_image_tensor(
                    orginal_image.crop((x1, y1, x2, y2)))
                input = __get_prompt_input(
                    image_size=1, type=type, multi_page=multi_page)
                result_prompt_inputs.append(input)
                result_prompt_images.append(image)
            else:
                orginal_image = Image.open(image_list)
                for box_item in box:
                    x1, y1, x2, y2 = box_item
                    image = __load_image_tensor(
                        orginal_image.crop((x1, y1, x2, y2)))
                    input = __get_prompt_input(
                        image_size=1, type=type, multi_page=multi_page)
                    result_prompt_inputs.append(input)
                    result_prompt_images.append(image)
        else:
            image_folder = image_list if image_list.endswith(
                "/") else image_list + "/"
            image_list = []
            for image_ext_name in image_ext_names:
                image_list.extend(
                    glob.glob(image_folder + '*' + image_ext_name))
            image_list = natsorted(image_list)
            for sub_image in image_list:
                image = __load_image_tensor(sub_image)
                input = __get_prompt_input(
                    image_size=1, type=type, multi_page=multi_page)
                result_prompt_inputs.append(input)
                result_prompt_images.append(image)
    elif isinstance(image_list, list):
        multi_page = isinstance(image_list[0], list)
        if multi_page is False:
            for sub_image in image_list:
                image = __load_image_tensor(sub_image)
                input = __get_prompt_input(
                    image_size=1, type=type, multi_page=multi_page)
                result_prompt_inputs.append(input)
                result_prompt_images.append(image)
        else:
            max_image_size = 1
            for sub_image_list in image_list:
                max_image_size = max(max_image_size, len(sub_image_list))

            for sub_image_list in image_list:
                sub_images = __load_image_tensor_list(sub_image_list)
                input = __get_prompt_input(image_size=len(
                    sub_image_list), type=type, multi_page=multi_page)
                result_prompt_inputs.append(input)
                result_prompt_images.append(sub_images)

    if len(result_prompt_inputs) == 0:
        raise ValueError("no image found, please check your image_file params")

    return result_prompt_inputs, result_prompt_images


def __split_image_datas(image_datas, image_sizes):
    result = []
    start_index = 0

    for condition in image_sizes:
        split_size = condition.item()
        if start_index + split_size <= len(image_datas):
            end_index = start_index + split_size
            sub_list = image_datas[start_index:end_index]
            result.append(sub_list)
            start_index = end_index
        else:
            break
    return result


# 模型推理时图片和文本的embedding嵌入
def merge_embeddings(embed_tokenizer, vision_tower_high, mm_projector_vary,
                     positions, input_ids, image_datas, image_sizes):
    inputs_embeds = embed_tokenizer(input_ids).cuda()
    if image_datas is None or image_sizes is None:
        return inputs_embeds

    # 同时兼容vllm 0.5.3/0.6.3.post1
    if isinstance(image_datas, Tensor):
        if image_datas.dim() == 5:
            P, B, D, W, H = image_datas.shape
            image_datas = image_datas.reshape(P * B, D, W, H)
        image_datas = __split_image_datas(image_datas, image_sizes)

    im_patch_token = 151859
    im_start_token = 151857
    im_end_token = 151858

    image_feature_list = []
    for sequence_images in image_datas:
        P, C, H, W = sequence_images.shape
        if P == 1:
            with torch.set_grad_enabled(False):
                cnn_feature = vision_tower_high(sequence_images)
                cnn_feature = cnn_feature.flatten(
                    2).permute(0, 2, 1)  # 256*1024

            image_feature = mm_projector_vary(cnn_feature)
            image_feature_list.append(image_feature)
        else:
            image_patches_features = []
            for image_patch in sequence_images:
                image_p = torch.stack([image_patch])
                with torch.set_grad_enabled(False):
                    cnn_feature_p = vision_tower_high(image_p)
                    cnn_feature_p = cnn_feature_p.flatten(2).permute(0, 2, 1)
                image_feature_p = mm_projector_vary(cnn_feature_p)
                image_patches_features.append(image_feature_p)
            image_feature = torch.cat(image_patches_features, dim=1)
            image_feature_list.append(image_feature)

    dummy_image_features = torch.zeros(
        256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    use_im_start_end = True
    new_input_embeds = []

    # 计算每个序列的范围
    zero_indices = torch.where(positions == 0)[0]
    zero_indices = torch.cat(
        [zero_indices, torch.tensor([len(positions)]).cuda()])
    sequence_ranges = [(start.item(), end.item())
                       for start, end in zip(zero_indices[:-1], zero_indices[1:])]

    # 前提是每个序列的内图片数量要相同才这么算的
    sequence_size = len(sequence_ranges)
    input_ids = input_ids.view(sequence_size, -1)
    NB, D = inputs_embeds.shape
    inputs_embeds = inputs_embeds.view(sequence_size, NB//sequence_size, D)

    for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_feature_list):
        if (cur_input_ids == im_patch_token).sum() == 0:
            cur_input_embeds = cur_input_embeds + \
                (0. * dummy_image_features).sum()
            new_input_embeds.append(cur_input_embeds)
            continue

        if use_im_start_end:
            if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                raise ValueError(
                    "The number of image start tokens and image end tokens should be the same.")

            image_start_tokens = torch.where(
                cur_input_ids == im_start_token)[0]
            for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                per_cur_image_features = per_cur_image_features.to(
                    device=cur_input_embeds.device)
                num_patches = per_cur_image_features.shape[0]

                if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                    raise ValueError(
                        "The image end token should follow the image start token.")

                cur_input_embeds = torch.cat(
                    (
                        cur_input_embeds[:image_start_token_pos + 1],
                        per_cur_image_features,
                        cur_input_embeds[image_start_token_pos +
                                         num_patches + 1:]
                    ),
                    dim=0
                )

            new_input_embeds.append(cur_input_embeds)
        else:
            raise NotImplementedError

    inputs_embeds = torch.stack(new_input_embeds, dim=0)

    if inputs_embeds.dim() == 3:
        B, N, D = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(N * B, D)

    return inputs_embeds
