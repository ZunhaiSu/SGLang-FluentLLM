import re
import json
import traceback

# Thinking model regex matching
def remove_think_content(text):
    # Find all content within <filename> tags
    answer_contents = re.sub(r"<think>\n.*?\n</think>", '', text, flags=re.DOTALL)
    think_contents = re.findall(r"<think>\n(.*?)\n</think>", text, flags=re.DOTALL)
    if think_contents != []:
        return answer_contents, think_contents[0]

    longcat_answer_contents = re.sub(r"(.*?)\n</longcat_think>\n", '', text, flags=re.DOTALL)
    longcat_think_contents = re.findall(r"(.*?)\n</longcat_think>\n", text, flags=re.DOTALL)
    if longcat_think_contents != []:
        return longcat_answer_contents, longcat_think_contents[0]
    
    return text, None

# Parameter conversion
def param_convert(param_name, param_info, required_params, indent_str=' '*8, is_return_type=False):
    # Whether required
    optional = "" if param_name in required_params else "?"

    param_type = param_info['type']
    assert param_type in ['integer', 'number', 'string', 'boolean', 'array', 'object'], f'invalid param type, {param_info}'
    if param_type == 'integer':
        ts_type = 'number'
    elif param_type == 'object':
        ts_params = []
        for param_name_, param_info_ in param_info['properties'].items():
            required_params_ = param_info.get('required', [])
            indent_str_ = indent_str + ' ' * 4
            param_str = param_convert(param_name_, param_info_, required_params_, indent_str_)
            ts_params.append(param_str)
        ts_params_str = ',\n'.join(ts_params)
        ts_type = '{\n' + ts_params_str + '\n' + indent_str + '}'
    elif 'enum' in param_info:
        ts_type = '"' + '" | "'.join(param_info['enum']) + '"'
    elif param_type == 'array':
        if 'items' in param_info:
            item_type = param_info['items']['type']
            if item_type == 'object':
                ts_params = []
                for param_name_, param_info_ in param_info['items']['properties'].items():
                    required_params_ = param_info['items'].get('required', [])
                    indent_str_ = indent_str + ' ' * 4
                    param_str = param_convert(param_name_, param_info_, required_params_, indent_str_)
                    ts_params.append(param_str)
                ts_params_str = ',\n'.join(ts_params)
                item_type = '{\n' + ts_params_str + '\n' + indent_str + '}'
            ts_type = item_type + '[]'
        else:
            ts_type = param_type
    else:
        ts_type = param_type

    if 'description' in param_info:
        ts_desc = param_info['description'].replace('\n', ' ')
    else:
        ts_desc = ''

    if 'example_value' in param_info:
        ts_example = param_info['example_value']
        ts_desc = f'{ts_desc}, example_value: {ts_example}'
    
    if is_return_type:
        param_str = f'{ts_type}; // {ts_desc}' if ts_desc != '' else f'{ts_type};'
    else:
        param_str = f'{indent_str}// {ts_desc}\n{indent_str}{param_name}{optional}: {ts_type}' if ts_desc != '' else f'{indent_str}{param_name}{optional}: {ts_type}'

    return param_str

# Convert functions to typescript types
def functions2typescript(functions):
    lst = []
    if not isinstance(functions, list):
        functions = [functions]
    for json_obj in functions:
        func_name = json_obj['name']
        func_desc = json_obj['description']
        func_params = json_obj['parameters']
        required_params = json_obj['parameters'].get('required', [])
        ts_params = []
        for param_name, param_info in func_params.get('properties', {}).items():
            param_str = param_convert(param_name, param_info, required_params)
            ts_params.append(param_str)
        ts_params_str = ',\n'.join(ts_params)
        returns_param_str = 'any;'
        if 'returns' in json_obj and json_obj['returns'].get('type', '') != '':
            param_info = json_obj['returns']
            returns_param_str = param_convert('returns', param_info, ['returns'], is_return_type=True)
        ts_func = f'\n    // {func_desc}\n    type {func_name} = (_:{{\n{ts_params_str}\n    }}) => {returns_param_str}'
        lst.append(ts_func)
    return '\n'.join(lst)


class LongcatPromptTemplate:
    # Special tokens used in prompt construction
    special_token_map = {
            'system': 'SYSTEM:',
            'user': 'USER:',
            'assistant': 'ASSISTANT:',
            'assistant_longcat_think': 'ASSISTANT:<longcat_think>\n',
            'tool': 'TOOL:',
            'function': '<|function|>\n',
            'multi_tool_use': '<|multi_tool_use|>\n',
            'code': '<|code|>\n',
            'retrieval': '<|retrieval|>\n',
            'files_start': '<|files_start|>\n',
            'files_end': '<|files_end|>\n',
    }

    # Template for functions prompt
    functions_prompt_template = '''
    ## functions

    namespace functions {

    %s

    }// namespace functions
    '''

    # Template for code prompt
    code_prompt_template = '''
    ## python

    When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment.
    Python will respond with the output of the execution or time out after 60.0 seconds.
    Internet access for this session is disabled.
    Do not make external web requests or API calls as they will fail.
    '''

    # Template for retrieval prompt
    retrieval_prompt_template = '''
    ## retrieval

    namespace retrieval {

        type search = (_: {
            query: string,
        }) => any;

    } // namespace retrieval
    '''

    # Template for multi-tool use prompt
    multi_tool_use_prompt_template = '''
    ## multi_tool_use

    // This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections. Only tools in the functions namespace are permitted.
    // Ensure that the parameters provided to each tool are valid according to that tool's specification.
    namespace multi_tool_use {

        // Use this function to run multiple tools simultaneously, but only if they can operate in parallel. Do this even if the prompt suggests using the tools sequentially.
        type parallel = (_: {
            // The tools to be executed in parallel. NOTE: only functions tools are permitted
            tool_uses: {
                // The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.
                recipient_name: string,
                // The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.
                parameters: object,
            }[],
        }) => any;

    } // namespace multi_tool_use
    '''


class PromptBuilder():
    def __init__(self):
        self.special_token_map = LongcatPromptTemplate.special_token_map
        self.functions_prompt = LongcatPromptTemplate.functions_prompt_template
        self.code_prompt = LongcatPromptTemplate.code_prompt_template
        self.retrieval_prompt = LongcatPromptTemplate.retrieval_prompt_template
        self.multi_tool_use_prompt = LongcatPromptTemplate.multi_tool_use_prompt_template

    def build_functions_prompt(self, functions):
        functions_prompt = functions2typescript(functions)
        functions_prompt = self.functions_prompt % functions_prompt
        return functions_prompt

    def build_tools_prompt(self, tools=None):
        tools_prompt = '# Tools\n'
        for tool in tools:
            if tool['type'] == 'function':
                if self.functions_prompt is not None and tool.get('function', None) is not None:
                    tools_prompt += self.build_functions_prompt(tool['function'])
                if self.multi_tool_use_prompt is not None:
                    tools_prompt += self.multi_tool_use_prompt
            elif tool['type'] == 'code_interpreter' and self.code_prompt is not None:
                tools_prompt += self.code_prompt
            elif tool['type'] == 'retrieval' and self.retrieval_prompt is not None:
                tools_prompt += self.retrieval_prompt
        return tools_prompt

    def func_check(self, function, functions_map):
        assert function['name'] in functions_map, f'invalid function name'
        function_define = functions_map[function['name']]
        args_json = json.loads(function['arguments'])
        function['arguments'] = json.dumps(args_json, ensure_ascii=False)
        required_params = function_define['parameters'].get('required', [])
        all_params = function_define['parameters'].get('properties', {})
        flag = True
        for k in args_json:
            if k not in all_params:
                flag = False
                print('undefined param: %s' % k)
                break
        for k in required_params:
            if k not in args_json:
                flag = False
                print('missing param: %s' % k)
                break
        assert flag, 'parameters error'
        return True

    def build_message_functions_prompt(self, function, content=None):
        prompt = self.special_token_map['function']
        prompt += '```typescript\n'
        if content is not None:
            prompt += '//%s\n' % content
        prompt += 'functions.%s(%s);' % (function['name'], function['arguments'])
        prompt += '\n```'
        return prompt

    def build_message_multi_tool_use_prompt(self, tool_calls, content=None, tools=None, functions_map={}):
        prompt = ''
        prompt += self.special_token_map['multi_tool_use']
        prompt += '```typescript\n'
        if content is not None:
            prompt += '//%s\n' % content
        tool_uses = []
        for tool_call in tool_calls:
            assert tool_call['type'] == 'function', 'Only tools in the functions namespace are permitted'
            function = tool_call['function']
            self.func_check(function, functions_map)
            function_name = 'functions.' + function['name']
            tool_use = {'recipient_name': function_name, 'parameters': function['arguments']}
            tool_uses.append(tool_use)
        tool_uses = {'tool_uses': tool_uses}
        tool_uses = json.dumps(tool_uses, ensure_ascii=False)
        prompt += 'multi_tool_use.parallel(%s);' % tool_uses
        prompt += '\n```'
        return prompt

    def build_message_retrieval_prompt(self, retrieval, content=None):
        prompt = self.special_token_map['retrieval']
        prompt += '```typescript\n'
        if content is not None:
            prompt += '//%s\n' % content
        prompt += 'retrieval.search(%s);' % retrieval
        prompt += '\n```'
        return prompt

    def build_message_code_prompt(self, code, content=None):
        prompt = ''
        prompt += self.special_token_map['code']
        if content is not None:
            prompt += content + '\n'
        prompt += '```python\n'
        prompt += code['input']
        prompt += '\n```'
        return prompt

    def build_target(self, message, tools=None):
        functions_map = {}
        if tools is not None:
            functions = [x['function'] for x in tools if x['type'] == 'function']
            for function in functions:
                functions_map[function['name']] = function

        message_prompt = ''
        if message.get('tool_calls', None) is not None and tools is not None:
            if len(message['tool_calls']) > 1:
                assert self.multi_tool_use_prompt is not None
                message_prompt += self.build_message_multi_tool_use_prompt(message['tool_calls'], message.get('content', None), tools, functions_map)
            else:
                tool_call = message['tool_calls'][0]
                assert tool_call['type'] in ['function', 'code', 'retrieval'], 'invalid tool type'
                if tool_call['type'] == 'function':
                    assert self.functions_prompt is not None
                    self.func_check(tool_call['function'], functions_map)
                    message_prompt += self.build_message_functions_prompt(tool_call['function'], message.get('content', None))
                elif tool_call['type'] == 'code':
                    assert self.code_prompt is not None
                    message_prompt += self.build_message_code_prompt(tool_call['code'], message.get('content', None))
                elif tool_call['type'] == 'retrieval':
                    assert self.retrieval_prompt is not None
                    message_prompt += self.build_message_retrieval_prompt(tool_call['retrieval'], message.get('content', None))
        else:
            message_prompt += message['content']
        return message_prompt

    def parse_target(self, resp_str, tool_choice='auto'):
        tool_choice_prefix = self.build_tool_choice_prefix(tool_choice)
        resp_str = tool_choice_prefix + resp_str
        content = None
        tool_calls = []
        if self.special_token_map['function'] in resp_str:
            resp_str = resp_str.split(self.special_token_map['function'] + '```typescript\n')[1].split('\n```')[0]
            cols = resp_str.split('functions.', 1)
            if cols[0].startswith('//'):
                content = cols[0].split('//')[1].strip('\n')
            cols = cols[1].split('(', 1)
            function_name = cols[0]
            function_args = cols[1].strip(');')
            function = {'type': 'function', 'function': {'name': function_name, 'arguments': function_args}}
            tool_calls.append(function)
        elif self.special_token_map['code'] in resp_str:
            cols = resp_str.split(self.special_token_map['code'])
            if cols[0] != '':
                content = cols[0]
            code_input = cols[1].split('```python\n')[1].split('\n```')[0]
            code = {'type': 'code', 'code': {'input': code_input}}
            tool_calls.append(code)
        elif self.special_token_map['retrieval'] in resp_str:
            resp_str = resp_str.split(self.special_token_map['retrieval'] + '```typescript\n')[1].split('\n```')[0]
            cols = resp_str.split('retrieval.search(', 1)
            if cols[0].startswith('//'):
                content = cols[0].split('//')[1].strip('\n')
            query_args = cols[1].strip(');')
            retrieval = {'type': 'retrieval', 'retrieval': query_args}
            tool_calls.append(retrieval)
        elif self.special_token_map['multi_tool_use'] in resp_str:
            resp_str = resp_str.split(self.special_token_map['multi_tool_use'] + '```typescript\n')[1].split('\n```')[0]
            cols = resp_str.split('multi_tool_use.parallel(', 1)
            if cols[0].startswith('//'):
                content = cols[0].split('//')[1].strip('\n')
            tool_uses = json.loads(cols[1].strip(');'))['tool_uses']
            for tool_use in tool_uses:
                function_name = tool_use['recipient_name'].split('functions.')[1]
                function_args = tool_use['parameters']
                function = {'type': 'function', 'function': {'name': function_name, 'arguments': function_args}}
                tool_calls.append(function)
        else:
            content = resp_str
        resp = {
            'role': 'assistant',
            'content': None,
            'tool_calls': None
            }
        if content is not None:
            answer_content, think_contents = remove_think_content(content)
            if think_contents is not None:
                resp['reasoning_content'] = think_contents
            resp['content'] = answer_content
        if len(tool_calls) > 0:
            resp['tool_calls'] = tool_calls
        return resp

    def build_message_prompt(self, message, round_num, tools=None):
        assert message['role'] in self.special_token_map, 'invalid role type'
        message_prompt = ""
        if message['role'] == 'user':
            message_prompt += f"[Round {round_num}] "
        message_prompt += self.special_token_map[message['role']]
        if message['role'] == 'system':
            message_prompt += message['content']
        elif message['role'] == 'user':
            if message.get('files', None) is not None:
                files_info = json.dumps(message['files'], ensure_ascii=False)
                message_prompt += self.special_token_map['files_start'] + files_info + self.special_token_map['files_end']
            message_prompt += message['content']
        elif message['role'] == 'assistant':
            message_prompt += self.build_target(message, tools)
        elif message['role'] == 'tool':
            tool_resp = {'content': message['content']}
            if 'name' in message.keys():
                tool_resp['name'] = message['name']
            tool_resp = json.dumps(tool_resp, ensure_ascii=False)
            message_prompt += tool_resp
        message_prompt += ' '
        return message_prompt

    def build_tool_choice_prefix(self, tool_choice):                                                                                                      
        prefix = ''                                                                                                                                       
        if type(tool_choice) == dict:                                                                                                                     
            tool_type= tool_choice['type']                                                                                                                
            assert tool_type in ['function', 'code', 'retrieval', 'multi_tool_use'], 'invalid tool type'                                                  
            prefix += self.special_token_map[tool_type]                                                                                                   
            if tool_type == 'function':                                                                                                                   
                prefix += '```typescript\nfunctions.' + tool_choice['function']['name']                                                                   
            elif tool_type == 'code':                                                                                                                     
                prefix += '```python\n'                                                                                                                   
            elif tool_type == 'multi_tool_use':                                                                                                           
                prefix += '```typescript\nmulti_tool_use.parallel'                                                                                        
            elif tool_type == 'retrieval':                                                                                                                
                prefix += '```typescript\nretrieval.search'                                                                                               
        return prefix     
    
    def build_input(self, messages, tools=None, tool_choice='auto', is_think_model=False):
        tools_prompt = ''
        if tools is not None and tool_choice is not None and tool_choice != 'none':
            tools_prompt += self.build_tools_prompt(tools)
        if tools_prompt != '':
            tools_prompt += '\n'
        round_num = 0
        messages_prompt = '# Messages\n\n'
        for message in messages:
            messages_prompt += self.build_message_prompt(message, round_num, tools)
            if message['role'] == 'user':
                round_num += 1
        if is_think_model:
            messages_prompt += self.special_token_map['assistant_longcat_think']
        else:
            messages_prompt += self.special_token_map['assistant']

        tool_choice_prefix = self.build_tool_choice_prefix(tool_choice)
        prompt = tools_prompt + messages_prompt + tool_choice_prefix
        return prompt

    def build_prompt(self, prompt):
        prompt_json = None
        try:
            prompt_json = json.loads(prompt)
            tools = prompt_json["tools"]
            messages = prompt_json["messages"]

            model_input = self.build_input(messages, tools, "auto")
            return model_input
        except:
            traceback.print_exc()
            print(json.dumps(prompt_json, ensure_ascii=False, indent=4))
            return prompt
