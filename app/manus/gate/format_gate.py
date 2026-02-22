from typing import Callable, Type, Any, get_type_hints, Union
from functools import wraps
from inspect import signature
import inspect

class FormatCheckError(Exception):
    """Exception raised when input/output format check fails"""
    pass
#支持的自动转换：
# 字符串转字典：str → dict (使用 json.loads)
# 字符串转列表：str → list (使用 json.loads)
# 字符串转整数：str → int
# 字符串转浮点数：str → float
# 任意类型转字符串：Any → str

def format_check(input_types: dict = None, output_type: Type = None):
    """
    Decorator to check function input/output types.
    If input_types and output_type are not provided, will try to parse from docstring.
    
    Args:
        input_types: Dictionary of parameter names to expected types
        output_type: Expected return type of the function
    """
    def decorator(func: Callable):
        # Use nonlocal to access outer scope variables
        nonlocal input_types, output_type
        
        # Parse docstring if input_types/output_type not provided
        input_types, output_type = _parse_docstring_if_needed(func, input_types, output_type)
        
        # Handle async functions
        if inspect.iscoroutinefunction(func):
            return _create_async_wrapper(func, input_types, output_type)
        
        return _create_sync_wrapper(func, input_types, output_type)
    return decorator

def _parse_docstring_if_needed(func: Callable, input_types: dict, output_type: Type) -> tuple[dict, Type]:
    """Parse docstring if input_types or output_type are not provided"""
    if input_types is None or output_type is None:
        doc = func.__doc__ or ""
        parsed_types = parse_docstring(doc)
        if input_types is None:
            input_types = parsed_types.get("input_types", {})
            print(f"[FormatCheck] Parsed input types from docstring: {input_types}")
        if output_type is None:
            output_type = parsed_types.get("output_type", None)
            print(f"[FormatCheck] Parsed output type from docstring: {output_type}")
    return input_types, output_type

def _create_sync_wrapper(func: Callable, input_types: dict, output_type: Type) -> Callable:
    """Create a synchronous wrapper function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = _bind_and_check_inputs(func, input_types, args, kwargs)
        if isinstance(bound_args, str):
            return bound_args  # Return error message if input check failed
        result = func(*bound_args.args, **bound_args.kwargs)
        output_error = _check_output(func, result, output_type)
        if output_error:
            return output_error  # Return error message if output check failed
        return result
    return wrapper

def _create_async_wrapper(func: Callable, input_types: dict, output_type: Type) -> Callable:
    """Create an asynchronous wrapper function"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        bound_args = _bind_and_check_inputs(func, input_types, args, kwargs)
        if isinstance(bound_args, str):
            return bound_args  # Return error message if input check failed
        result = await func(*bound_args.args, **bound_args.kwargs)
        output_error = _check_output(func, result, output_type)
        if output_error:
            return output_error  # Return error message if output check failed
        return result
    return async_wrapper

def _bind_and_check_inputs(func: Callable, input_types: dict, args: tuple, kwargs: dict) -> inspect.BoundArguments:
    """Bind arguments and check input types"""
    sig = signature(func)
    try:
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
    except Exception as e:
        return f"参数绑定失败: {str(e)}。请检查参数数量和类型是否正确。"

    if input_types:
        print(f"[FormatCheck] Checking input types for {func.__name__}")
        for param_name, expected_type in input_types.items():
            if param_name in bound_args.arguments:
                value = bound_args.arguments[param_name]
                if not isinstance(value, expected_type):
                    print(f"[FormatCheck] Attempting to convert parameter '{param_name}' "
                          f"from {type(value)} to {expected_type}")
                    try:
                        converted_value = convert_value(value, expected_type)
                        bound_args.arguments[param_name] = converted_value
                        print(f"[FormatCheck] Conversion successful for parameter '{param_name}'")
                    except FormatCheckError as e:
                        error_msg = (f"参数 '{param_name}' 类型不匹配: 期望 {expected_type}, 实际 {type(value)}。"
                                   f"建议: 请提供 {expected_type} 类型的值，或确保输入可以被自动转换为该类型。")
                        return error_msg
                else:
                    print(f"[FormatCheck] Parameter '{param_name}' type check passed: {expected_type}")
    return bound_args

def _check_output(func: Callable, result: Any, output_type: Type) -> str:
    """Check the output type of the function"""
    if output_type:
        print(f"[FormatCheck] Checking output type for {func.__name__}")
        if not isinstance(result, output_type):
            print(f"[FormatCheck] Attempting to convert return value from {type(result)} to {output_type}")
            try:
                result = convert_value(result, output_type)
                print(f"[FormatCheck] Return value conversion successful")
            except FormatCheckError as e:
                error_msg = (f"返回值类型不匹配: 期望 {output_type}, 实际 {type(result)}。"
                           f"建议: 请确保函数返回 {output_type} 类型的值，或确保返回值可以被自动转换为该类型。")
                return error_msg
        else:
            print(f"[FormatCheck] Return type check passed: {output_type}")
    return None

def convert_value(value, expected_type):
    """Attempt to convert value to expected type if possible"""
    try:
        # 处理可转换的类型
        if expected_type is dict and isinstance(value, str):
            import json
            print(f"[FormatCheck] Converting string to dict using json.loads")
            return json.loads(value)
        elif expected_type is list and isinstance(value, str):
            import json
            print(f"[FormatCheck] Converting string to list using json.loads")
            return json.loads(value)
        elif expected_type is int and isinstance(value, str):
            print(f"[FormatCheck] Converting string to int")
            return int(value)
        elif expected_type is float and isinstance(value, str):
            print(f"[FormatCheck] Converting string to float")
            return float(value)
        elif expected_type is str and not isinstance(value, str):
            print(f"[FormatCheck] Converting to string")
            return str(value)
        
        # 如果类型不匹配且无法转换
        if not isinstance(value, expected_type):
            raise FormatCheckError(f"无法将 {type(value)} 转换为 {expected_type}")
            
    except Exception as e:
        print(f"[FormatCheck] Conversion failed: {e}")
        raise FormatCheckError(
            f"无法将 {type(value)} 转换为 {expected_type}: {e}"
        )
    return value

def parse_docstring(doc: str) -> dict:
    """
    Parse function docstring to extract parameter types and return type.
    
    Args:
        doc: Function docstring
        
    Returns:
        dict: Dictionary with 'input_types' and 'output_type' keys
    """
    input_types = {}
    output_type = None
    
    lines = doc.split('\n')
    for line in lines:
        line = line.strip()
        # Parse parameter types
        if line.startswith('Args:'):
            continue
        if line.startswith('Returns:'):
            break
        if ':' in line and '(' in line and ')' in line:
            param_name = line.split(':')[0].strip().split(' ')[0]
            type_str = line.split('(')[1].split(')')[0].strip()
            try:
                input_types[param_name] = eval(type_str)
            except:
                continue
        
        # Parse return type
        if line.startswith('Returns:'):
            type_str = line.split('->')[1].strip() if '->' in line else line.split(':')[1].strip()
            try:
                output_type = eval(type_str.split(':')[0].strip())
            except:
                continue
    
    return {
        'input_types': input_types,
        'output_type': output_type
    }


@format_check()
def add(x, y):
    """
    Add two numbers.

    Args:
        x (int): First number
        y (int): Second number

    Returns:
        int: Sum of x and y
    """
    return x + y


result=add(1, 2)
print(result)