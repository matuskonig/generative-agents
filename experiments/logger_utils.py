import logging
from typing import override


class XMLExtraAdapter(logging.LoggerAdapter):
    @override
    def process(self, msg, kwargs):
        if "extra" in kwargs and len(kwargs["extra"]) > 0:
            content = "\n".join(
                f"<{key}>\n{value}\n</{key}>" for key, value in kwargs["extra"].items()
            )
            kwargs["extra"]["content"] = "\n" + content
        else:
            kwargs["extra"] = {"content": ""}
        return msg, kwargs


def get_xml_file_logger(file_name: str, level) -> logging.Logger:
    logger = logging.Logger(file_name, level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s%(content)s"
    )

    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return XMLExtraAdapter(logger)  # type: ignore
