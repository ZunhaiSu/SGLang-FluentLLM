import json
import ast
import logging
import re
from typing import List, Union, Literal


from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    _GetInfoFunc,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if arg_key not in tool.function.parameters["properties"]:
        return None
    return tool.function.parameters["properties"][arg_key].get("type", None)


def parse_arguments(json_value):
    try:
        try:
            parsed_value = json.loads(json_value)
        except:
            parsed_value = ast.literal_eval(json_value)
        return parsed_value, True
    except:
        return json_value, False


class LongCatXMLDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.bot_token = "<longcat_tool_call>"
        self.eot_token = "</longcat_tool_call>"
        self.func_call_regex = r"<longcat_tool_call>.*?</longcat_tool_call>"
        self.func_detail_regex = r"<longcat_tool_call>([^\n]*)\n(.*)</longcat_tool_call>"
        self.func_arg_regex = r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)</longcat_arg_value>"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(
        self,
        text: str,
        tools: List[Tool],
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none", "bypass_check"]]
    ) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
                pairs = re.findall(
                    r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)</longcat_arg_value>",
                    func_args,
                    re.DOTALL,
                )
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    arg_type = get_argument_type(func_name, arg_key, tools)
                    if arg_type != "string":
                        arg_value, is_good_json = parse_arguments(arg_value)
                    arguments[arg_key] = arg_value
                # construct match_result for parse_base_json
                final_match_result = {"name": func_name, "parameters": arguments}
                call = self.parse_base_json(final_match_result, tools, tool_choice)
                calls.extend(call)
                if func_name and arguments and len(call) == 0:
                    normal_text += match_result
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self,
        new_text: str,
        tools: List[Tool],
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none", "bypass_check"]]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        start = current_text.find(self.bot_token)
        if start == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                current_text = ""
            return StreamingParseResult(normal_text=current_text)
        # find ensures we find the first self.eot_token so there will be at most one tool_call in current_text[:end+len(self.eot_token)
        end = current_text.find(self.eot_token)
        if end != -1:
            # Initialize state if this is the first tool call
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
            # Ensure we have enough entries in our tracking arrays
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            result = self.detect_and_parse(
                current_text[: end + len(self.eot_token)], tools=tools, tool_choice=tool_choice
            )
            if result.calls:
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": result.calls[0].name,
                    "arguments": json.loads(result.calls[0].parameters),
                }
                self.streamed_args_for_tool[self.current_tool_id] = result.calls[
                    0
                ].parameters
                result.calls[0].tool_index = self.current_tool_id
                self.current_tool_id += 1
            self._buffer = current_text[end + len(self.eot_token) :]
            return result
        normal_text = current_text[:start]
        self._buffer = current_text[start:]
        return StreamingParseResult(normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()