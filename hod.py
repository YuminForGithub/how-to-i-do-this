#!/usr/bin/env python3
"""A small interpreter for the custom Brainfuck v2 esolang.

Ambiguity choices from the original notes:
* Main memory is a zero-filled, dynamically growing list. Moving left from
  address 0 leaves the pointer at 0 because negative list addresses are not
  part of the language's memory model.
* A `{...}` or `[...]` condition is tested after each pass through its body.
  At address 0, "the cell behind" means cell 0 rather than a negative address.
* `^` is an expression literal: a run of n carets evaluates to n. At top level
  it updates the interpreter's last value but does not overwrite a memory cell.
  It is most useful inside a jump, such as `` `^222` `` for address 8.
* `2` doubles the current cell as a command. Inside a jump expression it
  doubles the expression value, so `^^^22` evaluates to 12.
* Quotes do not support escapes. A string is ASCII only and ends at its next
  double quote. `%"text"` prints text directly; `"text"` stores it in memory.
"""

from __future__ import annotations

import sys
from collections.abc import Callable


class BrainFireError(Exception):
  """Raised when a program is not valid Brainfuck v2."""


class BrainFire:
  def __init__(self, input_reader: Callable[[], str] = input) -> None:
    self.input_reader = input_reader
    self.memory: list[int] = [0]
    self.pointer = 0
    self.last_value = 0
    self.output: list[str] = []

  def run(self, source: str) -> str:
    """Run source and return everything written by `.` `,`, and `%`."""
    self.memory = [0]
    self.pointer = 0
    self.last_value = 0
    self.output = []
    program = self._remove_comments(source)
    pairs = self._find_loop_pairs(program)
    pc = 0

    while pc < len(program):
      command = program[pc]

      if command.isspace():
        pc += 1
      elif command == ">":
        self.pointer += 1
        self._ensure_cell(self.pointer)
        pc += 1
      elif command == "<":
        self.pointer = max(0, self.pointer - 1)
        pc += 1
      elif command == "^":
        end = pc
        while end < len(program) and program[end] == "^":
          end += 1
        self.last_value = end - pc
        pc = end
      elif command == "|":
        self.last_value = self._cell()
        pc += 1
      elif command == "+":
        self._set_cell(self._cell() + 1)
        pc += 1
      elif command == "-":
        self._set_cell(self._cell() - 1)
        pc += 1
      elif command == "2":
        self._set_cell(self._cell() * 2)
        pc += 1
      elif command == "!":
        self._set_cell(0)
        pc += 1
      elif command == ";":
        self.pointer = 0
        pc += 1
      elif command == ".":
        value = self._cell()
        if not 0 <= value <= 127:
          raise BrainFireError(
            f"cannot print non-ASCII value {value} at address {self.pointer}"
          )
        self.output.append(chr(value))
        pc += 1
      elif command == ",":
        self.output.append(str(self._cell()))
        pc += 1
      elif command == "#":
        break
      elif command == "@":
        self._set_cell(self._read_integer())
        pc += 1
      elif command == '"':
        text, end = self._read_quoted(program, pc)
        for character in text:
          self._set_cell(ord(character))
          self.pointer += 1
          self._ensure_cell(self.pointer)
        pc = end
      elif command == "%":
        if pc + 1 >= len(program) or program[pc + 1] != '"':
          raise BrainFireError("`%` must be followed immediately by a quoted string")
        text, end = self._read_quoted(program, pc + 1)
        self.output.append(text)
        pc = end
      elif command == "`":
        end = program.find("`", pc + 1)
        if end == -1:
          raise BrainFireError("unterminated jump expression")
        address = self._evaluate_expression(program[pc + 1:end])
        if address < 0:
          raise BrainFireError("jump addresses cannot be negative")
        self.pointer = address
        self._ensure_cell(self.pointer)
        pc = end + 1
      elif command == "{":
        pc += 1
      elif command == "}":
        if self._behind_cell() != 1:
          pc = pairs[pc] + 1
        else:
          pc += 1
      elif command == "[":
        pc += 1
      elif command == "]":
        if self.pointer != 0:
          pc = pairs[pc] + 1
        else:
          pc += 1
      else:
        raise BrainFireError(f"unknown command {command!r} at character {pc}")

    return "".join(self.output)

  def _ensure_cell(self, address: int) -> None:
    while address >= len(self.memory):
      self.memory.append(0)

  def _cell(self) -> int:
    self._ensure_cell(self.pointer)
    return self.memory[self.pointer]

  def _set_cell(self, value: int) -> None:
    self._ensure_cell(self.pointer)
    self.memory[self.pointer] = value

  def _behind_cell(self) -> int:
    return self.memory[max(0, self.pointer - 1)]

  def _read_integer(self) -> int:
    try:
      return int(self.input_reader().strip())
    except ValueError as error:
      raise BrainFireError("`@` requires an integer console input") from error

  @staticmethod
  def _remove_comments(source: str) -> str:
    kept: list[str] = []
    in_comment = False
    in_string = False

    for character in source:
      if character == '"' and not in_comment:
        in_string = not in_string
        kept.append(character)
      elif character == "$" and not in_string:
        in_comment = not in_comment
      elif not in_comment:
        kept.append(character)

    if in_comment:
      raise BrainFireError("unterminated `$...$` comment")
    return "".join(kept)

  @staticmethod
  def _find_loop_pairs(program: str) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stacks = {"{": [], "[": []}
    closing = {"}": "{", "]": "["}
    in_string = False

    for index, character in enumerate(program):
      if character == '"':
        in_string = not in_string
      elif not in_string and character in stacks:
        stacks[character].append(index)
      elif not in_string and character in closing:
        opener = closing[character]
        if not stacks[opener]:
          raise BrainFireError(f"unmatched {character!r} at character {index}")
        start = stacks[opener].pop()
        pairs[start] = index
        pairs[index] = start

    if in_string:
      raise BrainFireError("unterminated string")
    for opener, stack in stacks.items():
      if stack:
        raise BrainFireError(f"unmatched {opener!r} at character {stack[-1]}")
    return pairs

  @staticmethod
  def _read_quoted(program: str, start: int) -> tuple[str, int]:
    end = program.find('"', start + 1)
    if end == -1:
      raise BrainFireError("unterminated string")
    text = program[start + 1:end]
    if any(ord(character) > 127 for character in text):
      raise BrainFireError("strings must contain ASCII characters only")
    return text, end + 1

  def _evaluate_expression(self, expression: str) -> int:
    """Evaluate `^`, `|`, and `2` in a backtick jump expression."""
    value: int | None = None
    index = 0

    while index < len(expression):
      character = expression[index]
      if character.isspace():
        index += 1
      elif character == "^":
        end = index
        while end < len(expression) and expression[end] == "^":
          end += 1
        value = end - index
        index = end
      elif character == "|":
        value = self._cell()
        index += 1
      elif character == "2":
        if value is None:
          raise BrainFireError("`2` in a jump must follow `^` or `|`")
        value *= 2
        index += 1
      else:
        raise BrainFireError(
          f"invalid jump expression character {character!r} at character {index}"
        )

    if value is None:
      raise BrainFireError("jump expression cannot be empty")
    return value


def main() -> None:
  source = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Program: ")
  try:
    output = BrainFire().run(source)
  except BrainFireError as error:
    print(f"Brainfuck v2 error: {error}", file=sys.stderr)
    raise SystemExit(1) from error
  print(output, end="")


if __name__ == "__main__":
  main()
