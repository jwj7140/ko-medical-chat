from typing import Iterator

import gradio as gr
import torch

from model import get_input_token_length, run

DEFAULT_SYSTEM_PROMPT = """아래는 매우 전문적인 의사와 환자의 진료 기록이다."""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 1500

DESCRIPTION = """
# Polyglot-Ko-Medical-Chat-5.8b

이 UI는 [Polyglot-Ko-Medical-Chat-5.8b](https://huggingface.co/squarelike/polyglot-ko-medical-chat-5.8b)에서 작동합니다.
AI에게 증상을 말하고 진료를 받아보세요!

이 모델은 의료적인 정확성을 보장하지 않습니다. 실제 증상이 있는 경우 반드시 의사와 상담하시길 바랍니다.
"""

if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'


def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message


def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return '', x


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Group():
        chatbot = gr.Chatbot(label='Chatbot')
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='Type a message...',
                scale=10,
            )
            submit_button = gr.Button('Submit',
                                      variant='primary',
                                      scale=1,
                                      min_width=0)
    with gr.Row():
        retry_button = gr.Button('🔄  Retry', variant='secondary')
        undo_button = gr.Button('↩️ Undo', variant='secondary')
        clear_button = gr.Button('🗑️  Clear', variant='secondary')

    saved_input = gr.State()

    with gr.Accordion(label='Advanced options', open=False):
        system_prompt = gr.Textbox(label='System prompt',
                                   value=DEFAULT_SYSTEM_PROMPT,
                                   lines=6)
        max_new_tokens = gr.Slider(
            label='Max new tokens',
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        temperature = gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.2,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )

    gr.Examples(
        examples=[
            '전 25세 여성입니다. 전봇대에 부딪친 후로 무릎이 너무 아파서 움직일 수가 없어요. 이건 무슨 병일까요?',
            '갑자기 배가 너무 아파요.',
            '몇 주 전에 감기에 걸렸었습니다. 열이나 다른 증상은 다 나았는데, 이상하게 기침만 아직도 멈추지 않아요.'
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=True,
    )
    
    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )

demo.queue(max_size=20).launch(server_name="0.0.0.0")
