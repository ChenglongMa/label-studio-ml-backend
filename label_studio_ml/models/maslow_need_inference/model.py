import logging
import json
import difflib
import re
import os
import requests
import pytesseract

from PIL import Image, ImageOps
from io import BytesIO
from typing import Union, List, Dict, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_random
from openai import OpenAI, AzureOpenAI

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag

logger = logging.getLogger(__name__)


@retry(wait=wait_random(min=5, max=10), stop=stop_after_attempt(6))
def chat_completion_call(messages, params, *args, **kwargs):
    """
    Request to OpenAI API (OpenAI, Azure)

    Args:
        messages: list of messages
        params: dict with parameters
           Example:
               ```json
              {
                "api_key": "YOUR_API_KEY",
                "provider": "openai",
                "model": "gpt-4",
                "num_responses": 1,
                "temperature": 0.7
                }```
    """
    provider = params.get("provider", OpenAIInteractive.OPENAI_PROVIDER)
    model = params.get("model", OpenAIInteractive.OPENAI_MODEL)
    if provider == "openai":
        client = OpenAI(
            api_key=params.get("api_key", OpenAIInteractive.OPENAI_KEY),
        )
        if not model:
            model = 'gpt-3.5-turbo'
    elif provider == "azure":
        client = AzureOpenAI(
            api_key=params.get("api_key", OpenAIInteractive.OPENAI_KEY),
            api_version=params.get("api_version", OpenAIInteractive.AZURE_API_VERSION),
            azure_endpoint=params.get('resource_endpoint', OpenAIInteractive.AZURE_RESOURCE_ENDPOINT).rstrip('/'),
            azure_deployment=params.get('deployment_name', OpenAIInteractive.AZURE_DEPLOYMENT_NAME)
        )
        if not model:
            model = 'gpt-35-turbo'
    elif provider == "ollama":
        client = OpenAI(
            base_url=params.get('base_url', OpenAIInteractive.OLLAMA_ENDPOINT),
            # required but ignored
            api_key='ollama',
        )
    else:
        raise

    request_params = {
        "messages": messages,
        "model": model,
        "n": params.get("num_responses", OpenAIInteractive.NUM_RESPONSES),
        "temperature": params.get("temperature", OpenAIInteractive.TEMPERATURE)
    }

    completion = client.chat.completions.create(**request_params)

    return completion


def gpt(messages: Union[List[Dict], str], params, *args, **kwargs):
    """
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    logger.debug(f"OpenAI request: {messages}, params={params}")
    completion = chat_completion_call(messages, params)
    logger.debug(f"OpenAI response: {completion}")
    response = [choice.message.content for choice in completion.choices]

    return response


class OpenAIInteractive(LabelStudioMLBase):
    """
    """
    OPENAI_PROVIDER = os.getenv("OPENAI_PROVIDER", "openai")
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    RESPONSE_PREFIX = os.getenv("RESPONSE_PREFIX", "response")
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    # if set, this prompt will be used at the beginning of the session
    DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs")
    NUM_RESPONSES = int(os.getenv("NUM_RESPONSES", 1))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")
    AZURE_RESOURCE_ENDPOINT = os.getenv("AZURE_RESOURCE_ENDPOINT", '')
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT")

    # TAGs `from_name` to be used for prediction
    NEED_CHOICES_TAG = os.getenv("NEED_CHOICES_TAG", "need")
    CONFIDENCE_RATING_TAG = os.getenv("CONFIDENCE_RATING_TAG", "confidence")
    COMMENTS_TEXT_AREA_TAG = os.getenv("COMMENTS_TEXT_AREA_TAG", "comments")

    def setup(self):
        if self.DEFAULT_PROMPT and os.path.isfile(self.DEFAULT_PROMPT):
            logger.info(f"Reading default prompt from file: {self.DEFAULT_PROMPT}")
            with open(self.DEFAULT_PROMPT) as f:
                self.DEFAULT_PROMPT = f.read()

    def _ocr(self, image_url):
        # Open the image containing the text
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = ImageOps.exif_transpose(image)

        # Run OCR on the image
        text = pytesseract.image_to_string(image)
        return text

    def _get_text(self, task_data, object_tag):
        """
        """
        data = task_data.get(object_tag.value_name)

        if data is None:
            return None

        if isinstance(object_tag, ImageTag):
            return self._ocr(data)
        elif isinstance(object_tag, ParagraphsTag):
            return json.dumps(data)
        else:
            return data

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        """Getting prompt values
        """
        if context:
            # Interactive mode - get prompt from context
            result = context.get('result')
            for item in result:
                if item.get('from_name') == prompt_tag.name:
                    return item['value']['text']
        # Initializing - get existing prompt from storage
        elif prompt := self.get(prompt_tag.name):
            return [prompt]
        # Default prompt
        elif self.DEFAULT_PROMPT:
            if self.USE_INTERNAL_PROMPT_TEMPLATE:
                logger.error('Using both `DEFAULT_PROMPT` and `USE_INTERNAL_PROMPT_TEMPLATE` is not supported. '
                             'Please either specify `USE_INTERNAL_PROMPT_TEMPLATE=0` or remove `DEFAULT_PROMPT`. '
                             'For now, no prompt will be used.')
                return []
            return [self.DEFAULT_PROMPT]

        return []

    def _match_choices(self, response: List[str], original_choices: List[str]) -> List[str]:
        # assuming classes are separated by newlines
        # TODO: support other guardrails
        matched_labels = []
        predicted_classes = response[0].splitlines()

        for pred in predicted_classes:
            scores = list(map(lambda line: difflib.SequenceMatcher(None, pred, line).ratio(), original_choices))
            matched_labels.append(original_choices[scores.index(max(scores))])

        return matched_labels

    def _find_choices_tag(self, object_tag):
        """Classification predictor
        """
        li = self.label_interface

        try:
            choices_from_name, _, _ = li.get_first_tag_occurence(
                'Choices',
                self.SUPPORTED_INPUTS,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(choices_from_name)
        except:
            return None

    def _find_textarea_tag(self, prompt_tag, object_tag):
        """Free-form text predictor
        """
        li = self.label_interface

        try:
            textarea_from_name, _, _ = li.get_first_tag_occurence(
                'TextArea',
                self.SUPPORTED_INPUTS,
                name_filter=lambda s: s != prompt_tag.name,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(textarea_from_name)
        except:
            return None

    def _find_rating_tag(self, object_tag):
        """Rating predictor
        """
        li = self.label_interface

        try:
            rating_from_name, _, _ = li.get_first_tag_occurence(
                'Rating',
                self.SUPPORTED_INPUTS,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(rating_from_name)
        except:
            return None

    def _find_prompt_tags(self) -> Tuple[ControlTag, ObjectTag]:
        """Find prompting tags in the config
        """
        li = self.label_interface
        prompt_from_name, prompt_to_name, value = li.get_first_tag_occurence(
            # prompt tag
            self.PROMPT_TAG,
            # supported input types
            self.SUPPORTED_INPUTS,
            # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))

        return li.get_control(prompt_from_name), li.get_object(prompt_to_name)

    def _validate_tags(self, *tags) -> None:
        """
        Validate tags
        """
        for tag in tags:
            if not tag:
                raise ValueError('No supported tags found')

    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict, labels: Optional[List[str]]) -> str:
        """
        """
        if self.USE_INTERNAL_PROMPT_TEMPLATE:
            norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt, labels=labels)
        else:
            norm_prompt = prompt.format(labels=labels, **task_data)

        return norm_prompt

    def _generate_response_regions(self, response: List[str], prompt_tag,
                                   need_choices_tag: ControlTag,
                                   confidence_rating_tag: ControlTag,
                                   comments_text_area_tag: ControlTag,
                                   llm_response_textarea_tag: ControlTag,
                                   prompts: List[str]) -> List:
        """
        """
        regions = []
        if not response or len(response) == 0:
            return regions
        pattern = r"Need:\s*(.+)\nConfidence:\s*([1-5])(?:\nComments:\s*(.*))?"
        match = re.search(pattern, response[0])
        if not match:
            logger.error(f"Response does not match expected format: {response}")
            return regions
        need = match.group(1)  # Extract <Need>
        confidence = match.group(2)  # Extract <Confidence (1–5)>
        comments = match.group(3) if match.group(3) else ""  # Extract <Additional comments or leave blank>

        if need_choices_tag:
            matched_labels = self._match_choices([need], need_choices_tag.labels)
            regions.append(need_choices_tag.label(matched_labels))
        if confidence_rating_tag:
            # confidence rating is a number between 1 and 5
            if confidence:
                matched_rating = int(confidence)
                if 1 <= matched_rating <= 5:
                    regions.append(confidence_rating_tag.label(matched_rating))
        if comments_text_area_tag:
            # comments are separated by newlines
            matched_comments = comments.splitlines()
            # Prefix each comment with 'LLM: '
            matched_comments = [f'LLM: {comment}' for comment in matched_comments]
            regions.append(comments_text_area_tag.label(text=matched_comments))

        if llm_response_textarea_tag:
            regions.append(llm_response_textarea_tag.label(text=response))

        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))

        return regions

    def _predict_single_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str,
                             need_choices_tag: ControlTag,
                             confidence_rating_tag: ControlTag,
                             comments_text_area_tag: ControlTag,
                             llm_response_textarea_tag: ControlTag,
                             prompts: List[str]) -> Dict:
        """
        """
        text = self._get_text(task_data, object_tag)  # NO USE.
        # Add {labels} to the prompt if choices tag is present
        labels = need_choices_tag.labels if need_choices_tag else None
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data, labels=labels)

        # run inference
        # this are params provided through the web interface
        response = gpt(norm_prompt, self.extra_params)
        regions = self._generate_response_regions(response, prompt_tag,
                                                  need_choices_tag,
                                                  confidence_rating_tag,
                                                  comments_text_area_tag,
                                                  llm_response_textarea_tag,
                                                  prompts)

        return PredictionValue(result=regions, score=0.1, model_version=str(self.model_version))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        """
        predictions = []

        # prompt tag contains the prompt in the config
        # object tag contains what we plan to label
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(context, prompt_tag)

        if prompts:
            prompt = "\n".join(prompts)

            need_choices_tag = self.label_interface.get_control(self.NEED_CHOICES_TAG)
            confidence_rating_tag = self.label_interface.get_control(self.CONFIDENCE_RATING_TAG)
            comments_text_area_tag = self.label_interface.get_control(self.COMMENTS_TEXT_AREA_TAG)

            llm_response_textarea_tag = self.label_interface.get_control(self.RESPONSE_PREFIX)
            self._validate_tags(need_choices_tag, confidence_rating_tag, comments_text_area_tag,
                                llm_response_textarea_tag)

            for task in tasks:
                # task['data'] contains the data for the task, it is a dict
                # Remove 'user_url' and 'item_url' from task['data'] if they exist
                task['data'].pop('user_url', None)
                task['data'].pop('item_url', None)
                # preload all task data fields, they are needed for prompt
                task_data = self.preload_task_data(task, task['data'])
                pred = self._predict_single_task(task_data, prompt_tag, object_tag, prompt,
                                                 need_choices_tag, confidence_rating_tag, comments_text_area_tag,
                                                 llm_response_textarea_tag, prompts)
                predictions.append(pred)

        return ModelResponse(predictions=predictions)

    def _prompt_diff(self, old_prompt, new_prompt):
        """
        """
        old_lines = old_prompt.splitlines()
        new_lines = new_prompt.splitlines()
        diff = difflib.unified_diff(old_lines, new_lines, lineterm="")

        return "\n".join(
            line for line in diff if line.startswith(('+',)) and not line.startswith(('+++', '---')))

    def fit(self, event, data, **additional_params):
        """
        """
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(data['annotation'], prompt_tag)

        if not prompts:
            logger.debug(f'No prompts recorded.')
            return

        prompt = '\n'.join(prompts)
        current_prompt = self.get(prompt_tag.name)

        # find substrings that differ between current and new prompt
        # if there are no differences, skip training
        if current_prompt:
            diff = self._prompt_diff(current_prompt, prompt)
            if not diff:
                logger.debug('No prompt diff found.')
                return

            logger.debug(f'Prompt diff: {diff}')
        self.set(prompt_tag.name, prompt)
        model_version = self.bump_model_version()

        logger.debug(f'Updated model version to {str(model_version)}')
