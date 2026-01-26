import inspect
import os
from pathlib import Path
import re
from enum import Enum
from typing import get_origin, get_args

# Specify class and function names to include
whitelist = {
    'src.qmap.benchmark': [
        "QMAPBenchmark",
        "DBAASPDataset",
        "Sample",
        "Bond",
        "Target",
        "HemolyticActivity"
    ],
    'src.qmap.toolkit': [
        'train_test_split',
        'compute_global_identity',
        'get_cache_dir',
        'compute_binary_mask',
        'read_fasta',
        'sequence_entropy',
        'Identity',
        'compute_maximum_identity',
        'create_edgelist',
        'build_graph',
        'leiden_community_detection',
        ]
}


def clean_signature(signature, function_name="function"):
    """
    Clean an inspect.signature object to create a readable function signature string.

    Args:
        signature: inspect.Signature object
        function_name: Name to use for the function (default: "function")

    Returns:
        str: Clean function signature string as it would appear in code
    """

    def clean_type_annotation(annotation):
        """Clean type annotation to get readable name without module paths"""
        if annotation == inspect.Parameter.empty:
            return None

        # Handle typing generics like List[str], Dict[str, int], Optional[str]
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            if args:
                arg_names = [clean_type_annotation(arg) for arg in args]
                origin_name = getattr(origin, '__name__', str(origin))
                return f"{origin_name}[{', '.join(filter(None, arg_names))}]"
            origin_name = getattr(origin, '__name__', str(origin))
            return origin_name

        # Handle regular types (int, str, float, custom classes, etc.)
        if hasattr(annotation, '__name__'):
            return annotation.__name__

        # Handle string annotations and edge cases
        annotation_str = str(annotation)
        if '.' in annotation_str:
            # Remove module path, keep just the class name
            return annotation_str.split('.')[-1]

        return annotation_str

    # Build parameter strings
    param_strings = []
    for param_name, param in signature.parameters.items():
        param_str = param_name

        # Add type annotation if present
        if param.annotation != inspect.Parameter.empty:
            clean_type = clean_type_annotation(param.annotation)
            if clean_type:
                param_str += f": {clean_type}"

        # Add default value if present
        if param.default != inspect.Parameter.empty:
            if isinstance(param.default, str):
                param_str += f' = "{param.default}"'
            elif param.default is None:
                param_str += " = None"
            else:
                param_str += f" = {param.default}"

        param_strings.append(param_str)

    # Build the function signature
    params_str = ", ".join(param_strings)
    signature_str = f"({params_str})"

    # Add return type if present
    if signature.return_annotation != inspect.Parameter.empty:
        return_type = clean_type_annotation(signature.return_annotation)
        if return_type:
            signature_str += f" -> {return_type}"

    signature_str += ":"

    return signature_str

def extract_params_and_clean_docstring(docstring):
    """
    This function extracts parameters from the docstring and returns:
    - the cleaned docstring (without parameter definitions),
    - a dictionary with parameter names as keys and their explanations as values.
    """
    # Regex to match the :param param_name: explanation pattern
    param_pattern = r':param ([A-Za-z_][A-Za-z0-9_]*):\s*(.*)'

    # Dictionary to store parameters and their explanations
    params_dict = {}

    # Find all matches
    matches = re.findall(param_pattern, docstring)

    # Extract parameters and their explanations
    for param_name, explanation in matches:
        params_dict[param_name] = explanation.strip()

    # Remove all parameter definitions from the docstring
    cleaned_docstring = re.sub(param_pattern, '', docstring)

    # Clean up any extra spaces or newlines that might remain
    cleaned_docstring = cleaned_docstring.strip()

    return cleaned_docstring, params_dict

def extract_returns_and_clean_docstring(docstring):
    """
    This function extracts returns explanations from the docstring and returns:
    - the cleaned docstring (without parameter definitions),
    - the return explanations as string.
    """
    if ":return:" in docstring:
        idx = docstring.find(":return:")

        exp = docstring[idx:].replace(":return:", "").strip()
        return docstring[:idx].strip(), exp
    else:
        return docstring, None

def extract_docstrings(package_path):
    # Create a markdown file to store the results
    markdown_path = Path("docs/references")
    if not os.path.exists(markdown_path):
        os.makedirs(markdown_path, exist_ok=True)

    # Walk through the package directory
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                module_name = file_path.replace('.py', '').replace('/', '.')

                # Import the module dynamically
                module = __import__(module_name, fromlist=[''])
                module_parent = ".".join(module_name.split('.')[:-1])
                if module_parent not in whitelist:
                    print("continuing with module:", module_parent)
                    continue
                # We do not want to write the file
                # md_file.write(f"## {module_name}\n\n")

                # Extract all classes and functions from the module
                for name, obj in inspect.getmembers(module):
                    if name not in whitelist[module_parent] or "__" in name:
                        continue
                    with open(f"{markdown_path}/{name}.md", 'w') as md_file:
                        md_file.write(f'*{module_parent}*\n')
                        # Check if it's a class
                        if inspect.isclass(obj) and issubclass(obj, Enum):
                            md_file.write(f"# Enum: `{name}`\n\n")
                            docstring = inspect.getdoc(obj) or "No documentation available"
                            md_file.write(f"**Description:** {docstring}\n\n")

                            # Extract enum members and their values
                            md_file.write(f"```python\n")
                            for enum_member in obj:
                                md_file.write(f"    {enum_member.name} = {enum_member.value}\n")
                            md_file.write(f"```\n\n")

                        elif inspect.isclass(obj):
                            md_file.write(f"# Class: `{name}`\n\n")
                            docstring = inspect.getdoc(obj) or "No documentation available"
                            md_file.write(f"**Description:** {docstring}\n\n")

                            # Extract methods of the class
                            for method_name, method_obj in inspect.getmembers(obj):
                                if method_name.startswith("_"):
                                    continue
                                if inspect.isfunction(method_obj):
                                    docstring = inspect.getdoc(method_obj) or "No documentation available"
                                    # Extract method signature and parameters
                                    signature = inspect.signature(method_obj)
                                    docstring, parameters = extract_params_and_clean_docstring(docstring)
                                    docstring, return_exp = extract_returns_and_clean_docstring(docstring)

                                    md_file.write(f"## Method: `{method_name}()`\n\n")
                                    md_file.write(
                                        f"```python\n{method_name}{clean_signature(signature)}\n```\n\n")
                                    md_file.write(f"**Description:** {docstring}\n\n")

                                    if len(parameters) > 0:
                                        md_file.write("**Parameters:**\n")
                                        for param, explanation in parameters.items():
                                            md_file.write(f"- `{param}`: {explanation}\n")
                                        md_file.write("\n")

                                    if return_exp and return_exp.strip() != "None":
                                        md_file.write("**Return:**\n")
                                        md_file.write(f"- {return_exp}\n")

                        # Check if it's a function (outside of class)
                        elif inspect.isfunction(obj):
                            docstring = inspect.getdoc(obj) or "No documentation available"

                            # Extract function signature and parameters
                            signature = inspect.signature(obj)
                            parameters = signature.parameters

                            md_file.write(f"# Function: `{name}()`\n\n")
                            md_file.write(
                                f"```python\n{name}{clean_signature(signature)}\n```\n\n")
                            md_file.write(f"**Description:** {docstring}\n\n")

                            # List the parameters and their types
                            if len(parameters) > 0:
                                md_file.write("**Parameters:**\n")
                                for param, explanation in parameters.items():
                                    md_file.write(f"- `{param}`: {explanation}\n")
                                md_file.write("\n")

    print(f"Docstrings extracted to {markdown_path}")


# Run the script on your package
extract_docstrings('src/qmap')