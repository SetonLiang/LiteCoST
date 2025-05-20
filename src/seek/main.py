from typing import Any, Iterable, List
import tiktoken
import json
import random

from openai import OpenAI 
import os
from dotenv import dotenv_values

# Schema-based Retrieval Agent
"""
CoT pipeline
1. query analysis: 
    1.1 Schema Extration: (Entity type, Entity name)
    1.2 Generate sub-qquestions 
2. chunking
3. find the chunks relevant to query
Does the provided context contain information needed to answer the query? Answer 'true' or 'false'
"""

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKENS=2048
random.seed(10086)

env_path = os.path.join('/data/liangzhuowen/projects/DocumentAI/llm', ".env")
config = dotenv_values(env_path)
# print(config)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1" 
openai_api_key = "sk-73d86EN1oRwFrgHa82C5Ed596837410eB6A6Ee3d011531C0"
openai_api_base = "https://api.gptapi.us/v1/chat/completions"
MODEL = "Qwen2.5-1.5B-Instruct"
GPT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base) 
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):    
    try:        
        response = client.chat.completions.create(           
            model=model,            
            messages=messages, 
            tools=tools,            
            tool_choice=tool_choice,       
            timeout=600 
        )        
        return response    
    except Exception as e:        
        print("Unable to generate ChatCompletion response")        
        print(f"Exception: {e}")        
        raise

tools = [
    {
        "type": "function",
        "description": "Determines if the provided text chunks contain information relevant to answering the user's question.",
        "function": {
            "name": "Query",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_context": {
                        "description": "Does the provided context contain information needed to answer the query? Answer 'true' or 'false'",
                        "title": "Keep Context",
                        "type": "boolean"
                    }
                },
                "required": ["keep_context"],
                "title": "Query",
                "type": "object",
                "additionalProperties": False
            }
        }
    }
]

def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def generate_prompt_Loong(item):

    replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
    prompt_template = item['prompt_template']
    # evidence = item['evidence']
    evidence = item['docs']
    
    for k, v in replace_dict.items():
        prompt_template = prompt_template.replace(k, v)

    if isinstance(evidence, list):  # 如果 evidence 是列表，则用换行符连接
        evidence = "\n".join(evidence)
    return prompt_template, evidence

# “二分法”递归拆分
def recursvie_split(section, max_tokens=MAX_TOKENS):
    """Split a section recursively based on token limits."""
    lines = section.split("\n")
    title = lines[0]
    content = "\n".join(lines[1:])

    half_length = len(content) // 2
    section_1 = f"**{title}**" + "\n" + content[:half_length]
    section_2 = f"**{title}**" + "\n" + content[half_length:]

    sections = [section_1, section_2]

    new_sections = []
    for sec in sections:
        if token_length(sec) > max_tokens:
            new_sections.extend(recursvie_split(sec))
        else:
            new_sections.append(sec)
    return new_sections

def split(section, max_tokens=MAX_TOKENS):
    """Directly split a section into multiple parts based on token limits."""
    lines = section.split("\n")
    title = lines[0]
    content = "\n".join(lines[1:])

    # 初始化
    sections = []
    current_section = f"**{title}**\n"
    current_tokens = token_length(current_section)

    for line in content.split("\n"):
        line_with_newline = line + "\n"
        line_tokens = token_length(line_with_newline)

        if current_tokens + line_tokens > max_tokens:
            sections.append(current_section.rstrip("\n"))  # 当前段落结束
            current_section = f"**{title}**\n" + line_with_newline  # 新段落
            current_tokens = token_length(current_section)
        else:
            current_section += line_with_newline
            current_tokens += line_tokens

    # 把最后一段加进去
    if current_section.strip():
        sections.append(current_section.rstrip("\n"))

    return sections


# 处理数据集的question
def process_Loong(file, output_folder):
    def generate_prompt_Loong(item):
        replace_dict = {"{question}": item['question'], "{instruction}": item['instruction']}
        prompt_template = item['prompt_template']
        # evidence = item['evidence']
        evidence = item['docs']
        
        for k, v in replace_dict.items():
            prompt_template = prompt_template.replace(k, v)

        if isinstance(evidence, list):  # 如果 evidence 是列表，则用换行符连接
            evidence = "\n".join(evidence)
        return prompt_template, evidence

    qa_datas = []
    result = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            qa_datas.append(record)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for record in qa_datas:
        record_id = record.get('id')
        if not record_id:
            print("Record missing id, skipping.")
            continue

        output_file = os.path.join(output_folder, f"{record_id}.json")
        # 如果文件已存在，则跳过该记录
        if os.path.exists(output_file):
            print(f"Record {record_id} already exists, skipping.")
            continue

        question, input_text = generate_prompt_Loong(record)

        chunks = split(input_text)
        print(len(chunks))
        rel_chunk = find_rel(question, chunks)
        record['evidence'] = rel_chunk

        # 将 record 写入到对应的文件中
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(record, out_f, ensure_ascii=False, indent=2)
        print(f"Record {record_id} processed and saved.")

    
     

def find_rel(query, chunks):
    results = []
    messages = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": "You are a AI assistant, your task is to determine if the provided text chunks contain information relevant to answering the user's question."},
            {"role": "user", "content": query},
            {"role": "user", "content": chunk}
        ]
        # print(messages)
        chat_response = chat_completion_request(
            messages, tools=tools, tool_choice={"type": "function", "function": {"name": "Query"}}
        )
        final_response = chat_response.choices[0].message
        keep_context = False
        # print(final_response)
        # 检查是否存在函数调用信息
        if final_response.tool_calls:
            try:
                tool_call = final_response.tool_calls[0]
                print(tool_call)
                args = tool_call.function.arguments
                # print(args)
                parsed_args = json.loads(args)
                # print(parsed_args, type(parsed_args))
                keep_context = parsed_args.get("keep_context", False)
            except Exception as e:
                print("Error parsing function call arguments:", e)
        else:
            # 如果没有函数调用，则尝试从 content 中解析（备用方案）
            try:
                args = final_response.content
                keep_context = args.get("keep_context", False)
            except Exception as e:
                print("Error parsing content:", e)
        # 如果当前 chunk 与查询相关，则添加到结果中
        if keep_context:
            # print(keep_context)
            results.append(chunk)
        # break
        
    return results


if __name__ == '__main__':
    qa_file = "/data/liangzhuowen/projects/DocumentAI/dataset/Loong-main/data/loong_process.jsonl"
    output_folder = "/data/liangzhuowen/projects/DocumentAI/dataset/Loong-main/data/loong_ours"

    process_Loong(qa_file, output_folder)



    
