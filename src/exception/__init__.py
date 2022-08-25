import os, sys

class CustomException(Exception):
    def __init__(self, error_msg:Exception, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = CustomException.custom_error_message(error_msg, error_detail)
    
    @staticmethod
    def custom_error_message(msg: Exception, detail:sys) -> str:
        '''
        returns a custom error message with filename, try/exception block line numbers and the message
        '''
        _, _, exec_try_block = detail.exc_info()
        exec_block_line_nbr = exec_try_block.tb_frame.f_lineno
        try_block_line_nbr = exec_try_block.tb_lineno
        file_name = exec_try_block.tb_frame.f_code.co_filename

        error_message = f'''
        Error occured in script: [ {file_name} ] at 
        try block line number: [{try_block_line_nbr}] and exception block line number: [{exec_block_line_nbr}] 
        error message: [{msg}]
        '''

        return error_message

    def __str__(self):
        # prints custom error message
        return self.error_msg

    def __repr__(self) -> str:
        return CustomException.__name__.str()