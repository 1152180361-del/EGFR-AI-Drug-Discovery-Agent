from abc import ABC, abstractmethod


class BaseTool(ABC):

    name = "base_tool"


    @abstractmethod
    def run(self, **kwargs):

        pass


    def info(self):

        return {

            "name": self.name,

            "description":
            "Computational biology tool"

        }
