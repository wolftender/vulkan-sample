#!/usr/bin/python3

import sys
import pathlib
import argparse
import string

def _convert_to_var_name(path: pathlib.Path) -> str:
    lower_ascii = set(string.ascii_lowercase)
    upper_ascii = set(string.ascii_uppercase)

    file_name = path.name
    var_name = ''
    next_upper = True

    for char in file_name:
        if char == '_':
            next_upper = True
        elif char in lower_ascii:
            if next_upper:
                var_name += char.upper()
            else:
                var_name += char
            next_upper = False
        elif char in upper_ascii:
            var_name += char
            next_upper = False
        else:
            var_name += '_'

    return var_name

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    input_file = pathlib.Path(args.input)
    output_file = pathlib.Path(args.output)

    if not input_file.exists() or not input_file.is_file():
        print(f'{input_file.resolve()} is not a valid input file', file=sys.stderr)
        return 1
    
    output_file.parent.mkdir(exist_ok=True)
    input_size = input_file.stat().st_size
    var_name = _convert_to_var_name(input_file)

    with open(input_file, mode='rb') as fin, open(output_file, mode='w') as fout:
        byte_num = 0
        col_width = 10
        tab_width = 4

        tab_str = ' ' * tab_width

        fout.write('#pragma once\n')
        fout.write('#include <array>\n#include <cstdint>\n\n')
        fout.write(f'constexpr std::array<uint8_t, {input_size}> k{var_name} = {{\n{tab_str}')

        while (byte := fin.read(1)):
            byte_num += 1
            hex_byte = '0x{:02x}'.format(byte[0])
            if (byte_num % col_width == 0):
                fout.write(hex_byte + ',\n' + tab_str)
            elif (byte_num == input_size):
                fout.write(hex_byte + '\n')
            else:
                fout.write(hex_byte + ', ')

        fout.write('};\n')

if __name__ == '__main__':
    sys.exit(run())