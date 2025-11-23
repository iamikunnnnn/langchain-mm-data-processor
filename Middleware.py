from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request: ToolCallRequest, handler) -> ToolMessage:
    """
    工具调用错误处理中间件：捕获执行异常，返回友好提示，避免原始错误信息泄露
    支持区分常见错误类型（参数错误、文件不存在、工具未找到等）
    """
    try:
        # 尝试执行工具调用
        return handler(request)
    except FileNotFoundError as e:
        # 文件相关错误（如 read_file 时文件不存在）
        return ToolMessage(
            content=f"❌ 工具调用失败：找不到指定文件，请检查文件路径是否正确（{str(e).split(':')[-1].strip()}）",
            tool_call_id=request.tool_call["id"],
            status="error"  # 标记状态为错误，帮助 Agent 识别
        )
    except ValueError as e:
        # 参数错误（如传入格式错误、缺失必填参数）
        return ToolMessage(
            content=f"❌ 工具调用失败：输入参数无效，请检查参数格式或补充必填信息（错误提示：{str(e)[:50]}）",
            tool_call_id=request.tool_call["id"],
            status="error"
        )
    except PermissionError as e:
        # 权限错误（如写文件时无权限）
        return ToolMessage(
            content=f"❌ 工具调用失败：无操作权限（如读写文件、访问资源），请检查权限设置",
            tool_call_id=request.tool_call["id"],
            status="error"
        )
    except Exception as e:
        # 其他未知错误（精简错误信息，避免泄露敏感细节）
        error_msg = str(e)[:80]  # 限制错误信息长度，防止冗余
        return ToolMessage(
            content=f"❌ 工具调用失败：未知错误，请稍后重试或联系管理员（错误摘要：{error_msg}...）",
            tool_call_id=request.tool_call["id"],
            status="error"
        )