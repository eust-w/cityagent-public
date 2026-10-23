"""Autonomous source-grounded reading through the actual TongAgents SDK.

The outer run is an append-only DAG of sessions and semantic reviews. A native
ReAct session chooses reading and workspace actions; no fixed chunk extraction
schedule stands in for the agent. Review batches bound context, not semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import threading
from typing import Any, Callable

from .document_model_gateway import (AuditedModelGateway, VerifiedGenerationLengthError,
    verified_incomplete_tool_terminal)
from .local_model_client import text_generation_parameters, validate_temperature
from .generation_stop_observer import generation_stopped
from .document_delivery_validation import validate_delivery_data
from .document_delivery_artifacts import artifact_page_matches
from .docx_writer import BASIC_FIELDS, ADVICE_FIELDS, DEFECT_FIELDS
from .evidence_manifest import render_spans
from .inference_core import _parse_report
from .run_state import atomic_json, json_digest
from .document_reading_recovery import (bounded_recovery, scan_workspace_retry, reading_length_attempt,
    LENGTH_RECOVERY_VERSION, LENGTH_RECOVERY_INSTRUCTION)
from .tong_react_adapter import run_react, session_boundary, wrap_tool_response, NATIVE_REVIEW_ADAPTER_VERSION


VERSION = "tongagents-document-workflow-bounded-semantic-review-v1"
PROMPT_VERSION = "source-grounded-autonomous-reader-missing-value-guidance-v1"
ROW_APPEND_VERSION = "explicit-single-new-row-v1"
WORKING_NOTES_VERSION = "explicit-model-notes-string-or-object-v1"
DRAFT_FEEDBACK_VERSION = "structural-draft-impact-feedback-v1"
REVIEW_VERSION = "bidirectional-semantic-review-v4"
REVERSE_LENGTH_VERSION = "verified-length-single-primary-bisection-v1"
NATIVE_REVIEW_VERSION = "stage-scheduled-native-text-review-v1"
POST_REVIEW_RECOVERY_VERSION = "current-passed-ready-delivery-feedback-v1"
MISSING_VALUE_GUIDANCE_VERSION = "explicit-null-existing-draft-api-guidance-v1"
MISSING_VALUE_GUIDANCE = (
    "缺失信息的写入操作：本次可用原文未提供的信息，不等于报告或现场没有该信息或病害；"
    "不能为了填空或满足绑定而新增‘无……’否定事实。未知值使用 JSON null 或空串，不用非空字符串‘未知’或‘无’占位。"
    "原文确有明确否定陈述时，仍可依据原文记录，并保留其适用对象与范围。"
    "当前 set_basic 即使 value 为 null，也要求非空且真实有效的 fact_ids；不能传空ID、漏掉该参数，或借用无关事实绕过校验。"
    "分批构建草稿时，可以暂不写未提供的基本信息字段。确需显式保存这些未知字段，且当前完整草稿能在本次输出额度内提交时，"
    "先用 read_workspace 及必要的草稿分页读取同一当前版本，保留已有全部有依据字段和明细行，再一并提交 draft、draft_fact_ids、draft_bindings。"
    "完整 draft 仍恰含基本信息对象、维修建议数组和病害明细数组；未知基本字段写 null 或空串，不给这些字段添加 binding，"
    "其余非空字段与每条明细行仍逐项绑定真实有效的事实或关系，并将这些ID列入 draft_fact_ids。"
    "不要混用完整 draft 与 draft_patch，不要为了此次写入漏掉、清空或截断已有内容。"
    "全稿较长时继续使用原有分批路径，不要求为了先填齐未知字段而改写全稿；这段说明不增加输出额度，也不替代长输出恢复。"
)
POST_REVIEW_RECOVERY_INSTRUCTION = (
    "当前绑定草稿已通过独立复核且交付就绪。先调用 read_workspace 查看既有草稿与 delivery 状态；"
    "没有新的原文证据时，不要仅为补齐空字段而重写，确认后调用 finish。"
    "若发现有原文依据的新事实或错误，可以显式修订，并重新调用 review_draft；修改后旧审核不再有效。"
    "此前工具错误已由当前通过的审核与交付状态解决，不作为待修任务。"
)
DELIVERY_FIELDS = {"基本信息": [name for name, _ in BASIC_FIELDS],
                   "维修建议": ADVICE_FIELDS, "病害明细": DEFECT_FIELDS}

SYSTEM_PROMPT = """你是桥梁检测报告阅读智能体，使用自有模型和 TongAgents 工具完成忠实抽取。
自由决定先读哪里、搜索什么、哪些年份/左右幅/构件需要关联，以及何时修订自己的理解。
任务是基本信息、维修建议、病害明细三部分；维修建议和病害原因/风险必须来自本报告。
未提供的信息保持未知；没有历史记录不等于没有病害，不能据此编造是否新增或发展程度。
是否新增与发展程度分别判断：前者是同一病害此前是否存在，后者是严重程度、范围或数量的前后变化。
未新增、前后病害名称相同，都不能单独证明未加重或无发展。原文没有发展评价或可比量测时保持未知，不用“无”代替缺失。
若根据同一对象的可比量测判断变化，事实标为 inferred，保存两次依据，并在输出中说明是量测比较推断，不能冒充报告明确结论。
区分原文明确陈述、专业推断、未确定和冲突。推断可存工作区，不能伪装成报告明确结论。
通过 read_outline/read_section/read_source/read_table_rows/search_in_document/get_neighbors 阅读。
长单元继续调用同一阅读工具和对象，程序按成功回执自动续读；切换章节或对象分别记录位置，需要从头重读时restart=true，不用抄写continuation或resume_section。
表头或注释超预算时，继续调用同一高层阅读工具，由程序按真实continuation读取完整上下文关系清单和原文，再读正文并返回章节；上下文元数据不等于原文覆盖，不要反复缩正文绕过上下文。
表格保留表头、注释，图片按需 read_image；不可判读的图片以 exclusions:[{image_id,reason}] 明示限制。
不要把调用工具、找到编号或读到摘要当成已经理解。不能用同名文字合并不同来源实例。
先读取原文，再写事实。原文不可变，跨章节桥名可继承，具体病害、时间、位置要核对对象。
read_workspace 读取当前 version 后，update_workspace(update={...}) 提交原生 JSON 对象（base_version 要用最新值）：
update 是对象，不是再次转义的 JSON 字符串。每次只提交一小组已读事实及必要的 draft_patch，成功保存后再继续，不要首次重写整份报告。
每次 update 顶层必须有 base_version；每条新事实必须有 id、value、source_refs、status。依赖和草稿绑定引用的 ID 必须已保存且有效，或在本次 update 中一并声明；不能先引用尚未创建的事实。工作区校验拒绝时整次更新均未保存；工具失败后按修复提示先 read_workspace 确认当前状态，再修正重试。
工作区较大时 read_workspace 只返回摘要与计数；用 read_workspace_page(view,limit,restart=false) 分页读取事实、草稿、问题及复核问题，同一view自动续读，切换view分别记录位置，需要从头重读时restart=true。
分页结果属于返回的 version；版本改变后重新读取，不拼接不同版本。历史保存在磁盘，不重复送入上下文。
facts:[{id,value,source_refs:[{evidence_id,start,end}],status:"explicit|inferred|uncertain|conflict",depends_on:[]}]
source_refs 直接复用工具返回的 evidence_id/start/end；完整原文标记 [E… chars 0:N] 可引用该单元的 0:N 范围。不要自行估算字串的字符偏移；需要补充依据时用 read_source 或 get_neighbors 取得原文并复用回执中的准确区间。引用范围可以包含必要上下文，是否支持事实由语义复核判断。
同一行的当前病害、前次状态及发展结论可分别来自不同段落；绑定支撑这些非空字段的全部事实/关系，不能只绑定一条病害名称事实就认为历史结论也已留存依据。
relations 使用 id,subject_id,predicate,object_id,source_refs,status,depends_on，可按需建立关系。
questions:[{id,question,status:"open|resolved|unknown|conflict",reason}]；未确定问题写明原因。
draft:{"基本信息":{字段:值},"维修建议":[{建议类别,建议内容,病害部位}],
"病害明细":[{病害部位,病害类型,病害描述,是否新增,上一次定检状态,发展程度}]}。
每次提交 draft 同时给 draft_fact_ids 和 draft_bindings，后者逐项绑定事实/关系 id：
{"基本信息.桥梁名称":["F1"],"维修建议.0":["F2"],"病害明细.0":["F3"]}。
事实更正会使依赖结论/草稿失效，请修订依赖记录并重新提交草稿。
没有任务相关信息的原文可 exclusions:[{evidence_id,start,end,reason}]，给明确理由，不要排除难读病害。
所有原文范围都要读过或说明排除，图片不清不能宣称没有病害。
准备交付时调用 review_draft，复核会同时检查输出依据和原文遗漏。
复核指出错误后回读、修订；通过后调用 finish。最终回答只说明状态，交付以工作区草稿为准。
知识库和自动策略进化将在后续阶段接入；当前不要假称调用了这些工具。
"""

SYSTEM_PROMPT += "\nreview_draft现在只排队冻结复核任务，不会在当前阅读工具里启动第二个SDK。收到queued后请结束本轮对话，说明已交接复核；这不是报告交付完成，不调用finish冒充通过。若继续修改草稿，已排队任务会因语义版本改变而作废。程序等当前SDK真正返回并结清后才独立复核，再在后续阅读会话提供结果。"

SYSTEM_PROMPT += "\n最终草稿只使用以下固定字段（其他分析保存在 facts/questions）；未知用 null 或空串，不补默认原因/风险。\n" + json.dumps(DELIVERY_FIELDS, ensure_ascii=False)
SYSTEM_PROMPT += "\n长草稿用 draft_patch 分多次提交，每次仍给最新 base_version。操作：set_basic(field,value,fact_ids)、append_row(section,row,fact_ids)、replace_row(section,index,row,fact_ids)、remove_row(section,index)。不得同时提交完整 draft。只更新必要字段/行，程序原子合成全稿；草稿内容仍由你决定。"
SYSTEM_PROMPT += '\nsearch_in_document 的 query 必须传原生JSON数组：一个短语用["左幅 1号梁"]，多个短语用["病害","新增"]进行字面OR搜索；不要把数组转义为字符串，空格拼接不是 OR 查询。'
SYSTEM_PROMPT += "\nread_source 通常只传 evidence_id，自动读取并按成功回执续读该单元；不要猜 end。需要上下文时用get_neighbors；原文引用的start/end是该单元内Unicode字符半开区间，不是全文位置，直接复用工具回执的准确区间。read_workspace_page 只需view/limit/restart，没有 draft 视图；草稿分别用 draft_basic/draft_advice/draft_defects，其他视图以 read_workspace 的 counts 为准。"
SYSTEM_PROMPT += "\n及时把自己的阅读进度和下一步存入 update.working_notes：可以是你写的字符串（程序原样保存为JSON对象{text:原字符串}），也可以是你写的JSON对象（内容原样保留）；其他类型不允许。存储对象的canonical JSON经UTF-8编码最多4096字节，覆盖旧笔记；{}清空。笔记可记已核事项、待读单元、待修补丁和下一步，只是阅读进度元数据，不作为原文证据或工程事实，不进入交付Word。原文阅读工具成功回包并登记阅读范围后，workspace.version可能已经推进；每次写入前先read_workspace取得当前version作为base_version，不沿用启动时或旧回包中的0，也不要猜测改成任何固定版本。每小组写入后更新笔记；预算将尽或工具报错时保存未完成事项，换会话先读当前工作区和笔记再继续，不从头重复搜索。"
SYSTEM_PROMPT += '\n新增病害或维修建议时优先使用单行格式 row_append：{"update":{"base_version":0,"row_append":{"fact_id":"F_NEW_EXAMPLE","section":"病害明细","row":{"病害描述":"占位：从本次已读原文核实的内容"},"source_refs":[{"evidence_id":"占位：实际工具编号","start":0,"end":1}],"status":"explicit"},"working_notes":{"next":"占位：下一步"}}}。这仅演示格式，不是答案或可直接执行的参数；实际版本、全新事实ID、字段值、编号、区间与status必须由你依据本次阅读明确给出。row_append恰有fact_id/section/row/source_refs/status五项，只新增一行；该行整体保存为同一事实，所有非空字段的依据都应在source_refs中保留，后续仍做语义复核。除base_version与可选working_notes，不得混用facts、relations、questions、draft_patch、draft或其他更新字段；需要多个独立事实或关系时仍用原格式。row只写一份，程序仅转换结构，不补值、不代替你判断。先成功保存一小行，再继续。'
SYSTEM_PROMPT += "\nupdate_workspace/read_workspace及review_draft阻塞回包的draft_invalidation_feedback是结构性诊断，不是原文事实或语义结论；其中列出更新版本、影响的绑定ID和当前草稿路径。更多项用read_workspace_page(view=draft_invalidations,limit=2,restart=false)自动分页及续读长项；版本变化时程序从头读取，不要混用不同version。重写已绑定fact时，即使value相同、只补source_refs，也须重新核对对应草稿。每次修复先read_workspace取当前version，再由你用set_basic或replace_row明确提供核实后的值和active fact_ids，或完整提交draft/draft_fact_ids/draft_bindings。程序不替你重发或修值。只改无关路径不会清除未确认路径提示；原实现可能把draft_valid整稿置true，本提示仍保留，这并不证明其他路径已重新核实。pending为诊断提示，不是新增审核阻塞或自动语义判断。"
SYSTEM_PROMPT += "\n图片工具的观察仍待核对，不是报告明确事实。自有视觉模型返回的图片事实引用必须带该次完整工具回包的 observation_id；同一图片不同问题或区域会生成不同观察，不能沿用旧审核。"
SYSTEM_PROMPT += "\n若存在未识别嵌入对象，先用 read_source_objects 分页列出，再用 read_source_object 取得逐个对象位置与完整上下文引用；必须用原文工具读完这些引用。对象清单和缓存图不证明已理解原生公式。草稿和必要阅读稳定后，逐个 propose_source_object(object_id,proposal={disposition,rationale,affected_output_paths,supporting_refs,context_refs,source_limitations})。原文引用格式仍为 evidence_id/start/end；不传 origin_call_ids 或 accepted。not_relied_on 表示相关输出有正文独立依据且没有依赖此对象；outside_task 须说明上下文为何与三表任务无关；unresolved_relevant/unclassified 保留未知并阻止完整交付。每个提案都由同一自有模型独立审核，不得因公式难读自动忽略；任何未解析限制均须保留。草稿/事实/已读范围变化会使旧提案失效，完成阅读和修订后重新提交。review_draft 包含独立对象审核；审核未通过须回读与修订。"

SYSTEM_PROMPT += "\n" + MISSING_VALUE_GUIDANCE
SYSTEM_PROMPT += (
    "\n局部保存与修复：draft_patch第一条set_basic或append_row可从空工作区建立三部分草稿。"
    "set_basic恰传op/field/value/fact_ids；append_row恰传op/section/row/fact_ids；"
    "replace_row恰传op/section/index/row/fact_ids；remove_row恰传op/section/index。"
    "只需补正已有条目引用时，用bind_output恰传op/path/fact_ids；它不改正文，不替你核验事实。"
    "工具拒绝整批更新时返回current_version、applied=false、具体missing_paths/invalid_ids/operation_index等。"
    "失败批次中的新事实与先前patch均未保存。先read_workspace核实，再按指出的位置提交小步修改；"
    "不要为修一条引用重新发送全文。完整draft仍必须同时给draft_fact_ids/draft_bindings，"
    "每个非空基本信息字段和每条明细行均保留真实支持事实的绑定。"
)

REVIEW_PROMPT = """你是独立复核本报告草稿的桥梁报告审阅者，使用上下文做语义判断。
FORWARD: 逐项核对输出是否由所附原文支持，理解跨段落、年份、左右幅、构件、位置关系。
REVERSE: 逐个原文范围核查是否存在应抽取但未列入相关草稿的基本信息、建议、病害或历史变化。
引用编号存在不等于支持，关键词不一致不等于错误。全报告桥名背景可跨段继承。
参考知识和图片观察不能替代原报告事实。没有历史记录不能推出新增/未新增或加重/未加重。
分别核对是否新增、前次状态与发展程度；未新增或前后名称相同不单独证明无发展/未加重。“无”不能默认为未知。
历史内容可能在别处有据：按需回读上下文，分别报告内容是否支持与已保存引用是否完整，不因缺少同段引用就把正确历史判成编造。
原文的细节须正确关联对象；同文不同来源可能是不同病害。未知应诚实保留。
related_output 只是由证据依赖找到的候选，不是完整草稿；不能因为它为空就判断遗漏。
draft_catalog 给出全稿值或分页入口；先理解全稿中已有内容，再判断当前原文事项是否遗漏。
FORWARD和REVERSE都可按需使用冻结全源目录review_read_source_catalog、原文搜索review_search_source、顺序章节review_read_source_section、范围review_read_source，以及全稿review_read_draft_page/review_search_draft/review_read_draft_item。当前引用缺少支持不等于全文无依据；请回读上下文，核不清楚则unresolved。
检索无命中不证明遗漏；需要时读全稿分页，核对年份、左右幅、构件、位置，不能为重复提及再加一行。
长稿分页时用 review_save_notes(notes_json) 单独保存简短JSON工作笔记：原文待核事项、已核草稿path与理由、下一页和未解决问题。
保存笔记不代表当前SDK立即压缩或暂停。新的复核会话会收到冻结任务和已存笔记，旧页仍可按同一冻结草稿重读。笔记不是原始证据，不确定时回读，不能凭记忆补事实。
related_output 的 fact_refs/source_refs 指向 related_facts/related_evidence；阅读这些完整原文和关系，核对跨段指代。
若需要未提供的上下文，列 unresolved_items，给明确原文位置和待查问题，不凭常识补齐。
review_assignment 固定本轮任务，绑定原文范围、输出项和冻结草稿。必须逐项核对全部分配内容，不能只看任务编号就通过。
仅返回JSON：{"passed":true或false,"checked_assignment_id":"原样复制review_assignment.id",
"checked_all_assigned":true或false,
"unsupported_claims":[],"missing_items":[],"conflicts":[],"unresolved_items":[]}。
checked_all_assigned=true 仅表示你已完成本轮全部范围的核查，不表示内容正确；发现的问题仍必须完整列出。
无法完成核查时 checked_all_assigned=false，不能宣称通过。不要重复抄写全部原文范围或字段路径；程序已保留分配清单。
问题项写 {path或evidence_id,reason,source_refs}。仅所有问题数组为空才 passed=true。
"""


REVIEW_PROMPT += "\n基本信息固定字段释义（沿用现有交付字段定义）：\n" + json.dumps(dict(BASIC_FIELDS), ensure_ascii=False)
REVIEW_PROMPT += (
    "\nFORWARD和REVERSE均须逐项分别核对：值是否有原文依据，以及原文字段的含义、来源角色是否支持目标输出字段。"
    "结合原标签、表头、相邻上下文和必要的跨段指代理解字段角色；名称近似、值相同或引用存在，不能单独证明字段角色等价。"
    "把原文信息用于另一目标字段时，也须核实原文是否支持这一关联。FORWARD检查输出值及其字段归属，"
    "REVERSE检查原文事项是否被正确字段覆盖；错误字段承载该值不能直接算作正确覆盖。"
    "无法核清时按既有协议列明依据不足、冲突或待核问题，保留原文位置与理由；未提供的信息保持未知，不能凭常识补齐。"
)

REVIEW_PROMPT += (
    "\n当前任务是完整报告审核的一批，只核对本批分配范围，不把整份报告重审一遍。"
    "其他草稿和全文可按需查询来核清本批关系；不要为了填充笔记重复确认已解决事项。"
    "工具调用较多时及时用review_save_notes保存已核范围、待核范围、具体待查问题和证据位置。"
    "达到会话预算后程序可以交接同一冻结任务，笔记不是原文或通过结论。"
    "本批完成仍须返回包含unsupported_claims、missing_items、conflicts、unresolved_items四个数组的完整JSON；"
    "每个数组即使为空也不得省略。未完成不能写checked_all_assigned=true。"
)


@dataclass(frozen=True)
class DocumentSettings:
    max_model_len: int = 65536
    output_tokens: int = 4096
    tool_tokens: int = 4096
    max_sessions: int = 12
    session_iterations: int = 8
    review_iterations: int = 32
    session_calls: int = 12
    report_timeout: float = 1200.0
    max_review_attempts: int = 2
    history_reserve_tokens: int = 4096
    max_model_calls: int = 192
    enable_thinking: bool = False
    temperature: float = 0.0
    compact_model_source: bool = False
    review_batch_items: int = 0
    dynamic_output_budget: bool = False
    workspace_bootstrap: bool = False
    single_document_agent: bool = False
    require_semantic_review: bool = True
    allow_quality_gaps: bool = False
    native_context_compression: bool = False
    compression_trigger_tokens: int = 196608

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in {"enable_thinking", "compact_model_source", "dynamic_output_budget", "workspace_bootstrap", "single_document_agent", "require_semantic_review", "native_context_compression", "allow_quality_gaps"}:
                if type(value) is not bool:
                    raise ValueError(f"{key} must be a boolean")
            elif key == "review_batch_items":
                if type(value) is not int or value < 0:
                    raise ValueError("review_batch_items must be a nonnegative integer")
            elif key == "temperature":
                validate_temperature(value)
            elif key == "report_timeout":
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("report_timeout must be finite and positive")
            elif type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.native_context_compression and (not self.single_document_agent or self.compression_trigger_tokens >= self.max_model_len):
            raise ValueError("Native compression requires single document Agent and trigger below actual context")
        if self.single_document_agent and (self.require_semantic_review or not self.dynamic_output_budget):
            raise ValueError("Single document Agent requires exact dynamic admission and explicit no-review policy")
        if not self.single_document_agent and not self.require_semantic_review:
            raise ValueError("No-review policy requires the explicit single document Agent mode")
        if self.dynamic_output_budget:
            if self.output_tokens != self.max_model_len:
                raise ValueError("Dynamic output_tokens is the context ceiling, not a fixed request budget")
            if self.tool_tokens + self.history_reserve_tokens + 1 >= self.max_model_len:
                raise ValueError("Tool/history page budgets must leave input room")
        elif self.output_tokens + self.tool_tokens + self.history_reserve_tokens >= self.max_model_len:
            raise ValueError("Output, tool and history reserves must leave input context")

    @property
    def planning_output_reserve(self):
        # Only a feasibility reserve. Actual outputs use the complete request
        # backend count, including SDK-created tools and historical messages.
        return 1 if self.dynamic_output_budget else self.output_tokens


class InitialSourceDeliveryError(ValueError):
    """First merged SDK request lost the supplied document/instruction binding."""


class DocumentWorkflowFailure(RuntimeError):
    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _generation_outcome_unknown(receipt: dict, model_calls: list[dict]) -> bool:
    """HTTP response headers alone do not settle an unfinished SSE stream."""
    if receipt.get("generation_attempted") is not True:
        return False
    if receipt.get("http_status") is None:
        return True
    # Older gateway checkpoints did not copy SSE termination fields. Recover
    # them from the bound actual call instead of treating HTTP 200 as DONE.
    actual = [call for call in model_calls if call.get("call_id") == receipt.get("call_id")]
    transport = actual[-1] if actual else receipt
    if "done_received" in transport and transport["done_received"] is not True:
        return not generation_stopped(transport)
    return False


def _recent_handler_errors(outcome: dict, enabled_tools: dict) -> list[dict]:
    """Carry only bounded adapter repair fields across native SDK sessions.

    The adapter maps exceptions to static repairs. Its argument/result receipts
    and extra diagnostics remain on disk; they are not recovery instructions.
    """
    recent = []
    calls = outcome.get("tool_calls", [])
    if not isinstance(calls, list):
        return recent
    for call in reversed(calls):
        if not isinstance(call, dict) or call.get("status") != "failed":
            continue
        name, error = call.get("name"), call.get("error")
        if not isinstance(name, str) or name not in enabled_tools or not isinstance(error, dict):
            continue
        if not all(isinstance(error.get(key), str) and error[key] for key in ("code", "action")):
            continue
        repair = {"name": name, "code": error["code"], "action": error["action"]}
        fields = error.get("fields", [])
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            continue
        if fields:
            repair["fields"] = fields.copy()
        raw_details = error.get("details")
        if (name == "update_workspace" and error["code"] in {
                "DRAFT_BINDINGS", "DRAFT_PATCH", "DRAFT_SHAPE", "DEPENDENCY_INVALID", "FACT_FIELDS"}
                and error.get("type") == "WorkspaceRepairError"
                and type(raw_details) is dict and raw_details.get("applied") is False):
            # Only program-owned structural coordinates cross a session. Never
            # copy arbitrary diagnostics, row values or a rejected draft.
            list_keys = ("missing_paths", "unknown_paths", "invalid_paths", "invalid_ids",
                         "invalid_binding_paths", "unlisted_ids", "empty_binding_paths",
                         "record_ids", "missing_fields", "invalid_fields", "unexpected_fields")
            details = {"applied": False}
            integer_keys = ("current_version", "operation_index", "row_count", "valid_index_count")
            integer_keys += tuple(key + suffix for key in list_keys for suffix in ("_count", "_omitted"))
            for key in integer_keys:
                value = raw_details.get(key)
                if type(value) is int and 0 <= value <= 2**31 - 1:
                    details[key] = value
            section = raw_details.get("section")
            if type(section) is str and section in {"基本信息", "维修建议", "病害明细"}:
                details["section"] = section
            for key in list_keys:
                value = raw_details.get(key)
                if (type(value) is list and len(value) <= 4
                        and all(type(item) is str and len(item) <= 96
                                and re.fullmatch(r"[\w.:-]+", item) is not None for item in value)):
                    details[key] = value.copy()
            if details:
                repair["details"] = details
        # Native repairs are already <= 1536 bytes. Bound restored receipts as
        # well without truncating a repair into an incomplete instruction.
        if len(_compact(repair).encode("utf-8")) > 2048:
            repair.pop("details", None)
            if len(_compact(repair).encode("utf-8")) > 2048:
                continue
        recent.append(repair)
        if len(recent) == 2:
            break
    return list(reversed(recent))


def _decode(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result
    value = json.loads(text, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _normalize_working_notes(update: dict) -> dict:
    """Wrap only model-authored string notes; original args remain upstream.

    The workspace still validates the stored object's finite JSON and canonical
    UTF-8 size. This metadata grants no fact, evidence or delivery authority.
    """
    if "working_notes" not in update:
        return update
    notes = update["working_notes"]
    if type(notes) is dict:
        return update
    if type(notes) is not str:
        from .document_workspace import WorkspaceError
        raise WorkspaceError("working_notes must be a string or finite JSON object")
    normalized = copy.deepcopy(update)
    normalized["working_notes"] = {"text": notes}
    return normalized


def _normalize_row_append(update: dict, existing_record_ids: set[str]) -> dict:
    """Expand a model-authored row once; never fill or judge its contents.

    The caller snapshots IDs before normalization. The unchanged strict
    apply_update base check then prevents a concurrent writer from making
    that snapshot stale and silently overwriting an intervening record.
    """
    if "row_append" not in update:
        return update
    from .document_workspace import WorkspaceError

    if set(update) - {"base_version", "row_append", "working_notes"}:
        raise WorkspaceError("row_append cannot mix with other update content fields")
    if type(update.get("base_version")) is not int or update["base_version"] < 0:
        raise WorkspaceError("row_append requires an explicit nonnegative integer base_version")
    row_append = update["row_append"]
    if (type(row_append) is not dict
            or set(row_append) != {"fact_id", "section", "row", "source_refs", "status"}):
        raise WorkspaceError("row_append requires exactly fact_id, section, row, source_refs and status")
    identifier = row_append["fact_id"]
    if type(identifier) is not str or not identifier.strip():
        raise WorkspaceError("row_append requires a nonempty fact_id")
    if identifier in existing_record_ids:
        raise WorkspaceError("row_append requires a new fact_id; use legacy updates for existing records")
    if row_append["section"] not in ("维修建议", "病害明细"):
        raise WorkspaceError("Row operations require section 维修建议 or 病害明细")
    if type(row_append["row"]) is not dict:
        raise WorkspaceError("Draft row must be an object")
    if type(row_append["source_refs"]) is not list:
        raise WorkspaceError("source_refs must be a list")
    if row_append["status"] not in ("explicit", "inferred", "uncertain", "conflict"):
        raise WorkspaceError("Invalid evidence record status")
    result = {
        "base_version": update["base_version"],
        "facts": [{"id": identifier, "value": copy.deepcopy(row_append["row"]),
                   "source_refs": copy.deepcopy(row_append["source_refs"]),
                   "status": row_append["status"]}],
        "draft_patch": [{"op": "append_row", "section": row_append["section"],
                         "row": copy.deepcopy(row_append["row"]), "fact_ids": [identifier]}],
    }
    if "working_notes" in update:
        result["working_notes"] = copy.deepcopy(update["working_notes"])
    return result


class DocumentWorkflow:
    def _reader_system_prompt(self):
        if not self.settings.single_document_agent:
            return SYSTEM_PROMPT
        lines = []
        for line in SYSTEM_PROMPT.splitlines():
            if (line.startswith("准备交付时调用 review_draft")
                    or line.startswith("复核指出错误后")
                    or line.startswith("review_draft现在")
                    or line.startswith("若存在未识别嵌入对象")):
                continue
            line = line.replace("问题及复核问题", "问题（本轮无复核）")
            line = line.replace("是否支持事实由语义复核判断", "是否支持事实由你依据完整上下文判断")
            line = line.replace("后续仍做语义复核", "内容判断由你依据原文完成")
            line = line.replace("及review_draft阻塞回包", "阻塞回包")
            line = line.replace("不能沿用旧审核", "不能沿用不相关的观察")
            line = line.replace("换会话先读当前工作区和笔记再继续，不从头重复搜索", "在当前会话继续；真实失败时保存已有进度，本轮不自动重开")
            lines.append(line)
        lines.append("输出协议：正在工作的回合直接调用工具；普通正文只用于简短说明，不能把报告写在普通正文里。报告通过update_workspace分批保存。普通正文超过4096字符会被视为操作回合协议超限并取消；此限制不含思考和工具JSON参数，也不是报告长度上限。finish成功后给简短真实最终状态，不能用空回答结束。")
        lines.append("推进交付：首次保存前先read_workspace取得真实version，全文已传入会更新阅读记录，不能假定版本仍为0。工具成功回执中的version才是下一次base_version。对已经确定要输出的病害或维修建议，在同一个update中提交facts和draft_patch的append_row；现有事实可直接绑定，不必再次改写笔记。已有行用replace_row修订，不重复append。未确定的语义问题存questions，不为凑表行编造。")
        lines.append("区分笔记与交付：工作区facts数不是表格行数。查看实际draft_counts；若有未绑定事实，用read_workspace_page(view=unbound_facts或unbound_relations,limit=4,restart=false)核对哪些候选应该写入病害/建议表，哪些只是背景。未绑定并不等于遗漏，关联和去重由你依据原文决定。不要只改基本信息或重复保存相同值。明确条目尽早落稿，再按需回读疑点。完成条目整理后再inspect_delivery，不能把空表预览当作完成。")
        lines.append("本轮没有独立语义复核流程。一个文档使用一个持续的 TongAgents Agent、一个上下文和独立工作区。阅读和保存后，由你自查实际导出的文件，不创建复核Agent，也不得声称独立复核通过。")
        lines.append("基本信息固定字段含义与导出器相同：" + _compact(dict(BASIC_FIELDS)))
        lines.append("基本信息的‘建议’是建议分类统计，‘建议统计’是明细条目统计，不是维修建议正文；具体建议写维修建议行。分类和统计须有完整依据，无法确定类别时保持未知，不猜统计。是否新增不等于是否加重；发展程度没有逐对象评价或可比量测时留空，专业推断必须标明。")
        lines.append("当前病害表没有单独的成因列。报告明确说明的成因写入对应对象的‘病害描述’，保留原因段的原文依据；专业推断明确标为推断，不能用常识或知识库补报告未说明的原因。自查时核对报告给出的原因是否完整关联到对应对象，未说明则保持未知。")
        lines.append("交付前先调用 inspect_delivery(view='summary',limit=4,restart=false)，然后按basic/advice/defects读完实际Word内容；重复调用同一view自动继续下一页，换view或文件内容变化自动从头开始，空表也读取确认，需要重读时restart=true。核对实际工具保存成功、原文支持、字段归属和遗漏；只检查摘要或工作区不能算检查交付文件。发现错漏回读原文并修订，再重新读取已改变的导出文件。仅阅读进度或导航笔记更新不改变同一Word的已读状态；事实、关联、出处、绑定、未决问题或排除范围改变后，你仍需重新作出内容自查判断。")
        lines.append("确认当前导出后调用 finish(self_check={inspected_export:true,complete:true,basis:简短公开自查依据,remaining_issues:[]})，不用抄写工作区版本或文件哈希。程序只绑定你这次请求确实收到并读完的当前Word工具回执；文件有修改时旧检查不能用于新版，须重新读取新版。不能完成时complete:false并列remaining_issues，不伪造成功。程序仅核格式、文件、分页覆盖、版本和模型调用来源；依据和完整性由你判断，不是程序语义审核。finish成功后再给出实际最终状态。")
        lines.append("局部会话次数不用于阶段交接。必须在当前会话完成；真实模型异常、生成截断、历史丢失或上下文容量边界会明确失败并保存进度，本轮不隐式压缩、不自动新会话或重试。未解析嵌入内容保留限制并阻止完整交付，不能假装完成。最终回答只说明实际状态，交付以已保存草稿为准。")
        if self.settings.native_context_compression:
            lines.append('上下文达到本轮验证阈值时，由原生TongAgents ContextCompressor调用自有4B模型保存语义笔记；程序先保存准确工具/工作区状态和完整旧历史，再原子替换当前Agent的活动Memory。笔记不是原文依据。必要内容继续用原工具回读；这不创建新Agent，不改变事实或文件检查记录。SDK解析JSON失败时会收到未执行、未保存的具体失败回包，请在同一Agent中重新生成小步合法参数。真实上下文/网络异常或压缩失败不会自动重开，保留未完成文件。')
        return "\n".join(lines)

    def __init__(self, manifest: dict, client, react_config: dict, directory: Path,
                 *, settings: DocumentSettings | None = None, source_path=None,
                 vision_backend=None, react_runner: Callable = run_react, run_identity: dict | None = None,
                 external_deadline: float | None = None, vision_backend_factory=None,
                 explicit_operator_resume: bool = False, compression_client=None, material_index=None):
        from .document_tools import DocumentTools
        from .document_workspace import DocumentWorkspace

        if external_deadline is not None and (type(external_deadline) not in (int, float)
                or not math.isfinite(external_deadline) or external_deadline <= 0):
            raise ValueError("External report deadline must be a positive finite monotonic timestamp")
        self.manifest, self.client = manifest, client
        self.material_index = material_index
        self.react_config = dict(react_config)
        self.settings = settings or DocumentSettings()
        self.compression_client = compression_client
        self.native_compression = None
        if self.settings.native_context_compression:
            if (compression_client is None or getattr(compression_client, 'dynamic_output_budget', None) is not True
                    or getattr(compression_client, 'enable_thinking', None) is not True
                    or getattr(compression_client, 'max_model_len', None) != self.settings.max_model_len):
                raise ValueError('Native compression requires pinned own thinking summary model with exact dynamic budget')
        elif compression_client is not None:
            raise ValueError('Summary client supplied while native compression is disabled')
        if type(explicit_operator_resume) is not bool or (explicit_operator_resume and not self.settings.single_document_agent):
            raise ValueError("Explicit operator resume must be boolean and only applies to single document Agent mode")
        self.explicit_operator_resume = explicit_operator_resume
        if getattr(client, "dynamic_output_budget", False) is not self.settings.dynamic_output_budget:
            raise ValueError("Workflow and client dynamic budget policies must agree")
        if self.settings.dynamic_output_budget and (vision_backend is not None or vision_backend_factory is not None):
            raise ValueError("Dynamic text policy does not replace exact multimodal native admission")
        client_thinking = getattr(client, "enable_thinking", False)
        if type(client_thinking) is not bool or client_thinking is not self.settings.enable_thinking:
            raise ValueError("Workflow and own-model client thinking settings must agree")
        if ("enable_thinking" in self.react_config
                and self.react_config["enable_thinking"] is not self.settings.enable_thinking):
            raise ValueError("Workflow and Agent thinking settings must agree")
        self.react_config["enable_thinking"] = self.settings.enable_thinking
        client_temperature = validate_temperature(getattr(client, "temperature", 0))
        if client_temperature != self.settings.temperature:
            raise ValueError("Workflow and own-model client temperature settings must agree")
        if ("temperature" in self.react_config
                and validate_temperature(self.react_config["temperature"]) != self.settings.temperature):
            raise ValueError("Workflow and Agent temperature settings must agree")
        self.react_config["temperature"] = self.settings.temperature
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.react_runner = react_runner
        if self.react_config.get("model") != client.model:
            raise ValueError("Agent and evidence reviewer must use the same pinned own model")
        if self.settings.max_model_len != client.max_model_len:
            raise ValueError("Workflow and own-model context limits must agree")
        identity = {
            "workflow": VERSION, "source": json_digest(manifest), "model": client.model,
            "upstream": client.base_url, "settings": asdict(self.settings),
            "system_prompt_sha256": json_digest(self._reader_system_prompt()), "review_prompt_sha256": json_digest(REVIEW_PROMPT),
            "versions": {"prompt": PROMPT_VERSION, "review": REVIEW_VERSION,
                         "row_append": ROW_APPEND_VERSION,
                         "working_notes": WORKING_NOTES_VERSION,
                         "draft_feedback": DRAFT_FEEDBACK_VERSION,
                         "reading_length": LENGTH_RECOVERY_VERSION,
                         "reverse_length": REVERSE_LENGTH_VERSION,
                         "native_review": NATIVE_REVIEW_VERSION,
                         "post_review_recovery": POST_REVIEW_RECOVERY_VERSION,
                         "missing_value_guidance": MISSING_VALUE_GUIDANCE_VERSION,
                         "missing_value_guidance_sha256": json_digest(MISSING_VALUE_GUIDANCE),
                         "post_review_recovery_instruction_sha256": json_digest(POST_REVIEW_RECOVERY_INSTRUCTION),
                         "native_review_adapter": NATIVE_REVIEW_ADAPTER_VERSION,
                         "reading_length_prompt_sha256": json_digest(LENGTH_RECOVERY_INSTRUCTION)},
            "run_identity": copy.deepcopy(run_identity or {}),
        }
        if material_index is not None:
            material_index.validate_manifest(manifest)
            identity["material_identity"] = copy.deepcopy(material_index.identity)
        if vision_backend_factory is not None:
            if vision_backend is not None or not callable(vision_backend_factory):
                raise ValueError("Select one bound vision backend or factory")
            factory_identity = getattr(vision_backend_factory, "identity", None)
            if not isinstance(factory_identity, dict):
                raise ValueError("Vision factory needs an immutable execution identity")
            identity["vision_backend"] = copy.deepcopy(factory_identity)
        self.source_objects = None
        source_catalog = None
        if self.settings.require_semantic_review and manifest.get("unverified_content") and source_path is not None:
            from .source_object_catalog import build_source_object_catalog
            from .document_source_objects import VERSION as source_version
            from .source_object_review import REVIEW_SYSTEM_PROMPT as source_prompt, PROTOCOL as source_protocol
            source_catalog = build_source_object_catalog(Path(source_path), manifest)
            identity["source_objects"] = {"catalog_sha256": source_catalog["catalog_sha256"],
                "controller": source_version, "review": source_protocol,
                "review_prompt_sha256": json_digest(source_prompt)}
        self.identity = identity
        self.trace_path = self.directory / "document_workflow.json"
        self.nodes = []
        self.sessions = []
        self.review_sessions = []
        self.pending_review = None
        self.handoff_events = []
        self.reading_terminal_attempts = []
        self.review_attempts = 0
        self.total_spent_seconds = 0.0
        self.model_calls_before_resume = []
        self.model_dispatches = []
        self.uncertain_budget_seconds = 0.0
        previous = {}
        if self.trace_path.exists():
            previous = json.loads(self.trace_path.read_text())
            if previous["identity"] != identity:
                raise ValueError("Workflow identity changed; cannot resume")
            self.nodes = previous.get("dag_nodes", [])
            self.sessions = previous.get("sessions", [])
            self.review_sessions = copy.deepcopy(previous.get("review_sessions", []))
            self.pending_review = copy.deepcopy(previous.get("pending_review"))
            self.handoff_events = copy.deepcopy(previous.get("handoff_events", []))
            self.reading_terminal_attempts = copy.deepcopy(previous.get("reading_terminal_attempts", []))
            self.review_attempts = previous.get("review_attempts", 0)
            self.total_spent_seconds = previous.get("spent_seconds", 0.0)
            self.model_calls_before_resume = previous.get("model_calls", [])
            self.model_dispatches = copy.deepcopy(previous.get("model_dispatches", []))
            terminal = {r.get("request_id"): r for r in previous.get("gateway_receipts", [])
                        if r.get("request_id")}
            for dispatch in self.model_dispatches:
                receipt = terminal.get(dispatch["request_id"])
                if receipt is not None:
                    dispatch.update(status=("outcome_unknown" if _generation_outcome_unknown(
                                            receipt, self.model_calls_before_resume) else "settled"),
                                    receipt_status=receipt["status"],
                                    generation_attempted=receipt.get("generation_attempted", False),
                                    call_id=receipt.get("call_id"))
                elif dispatch.get("status") in {"pending", "outcome_unknown"}:
                    # A lost process cannot establish whether the server ran
                    # its request. Preserve that uncertainty, reserve the
                    # remaining deadline, and do not silently replay it.
                    dispatch["status"] = "outcome_unknown"
                    self.uncertain_budget_seconds = max(
                        self.uncertain_budget_seconds,
                        dispatch.get("remaining_report_seconds", 0.0))
        if (self.settings.single_document_agent and self.sessions and previous.get("status") != "completed"
                and not self.explicit_operator_resume):
            raise ValueError("A prior single Agent session exists; explicit operator resume into a new context is required")
        self._invocation_session_start = len(self.sessions)
        self.started = time.monotonic()
        self.deadline = self.started + max(0.0, self.settings.report_timeout
                                          - self.total_spent_seconds - self.uncertain_budget_seconds)
        if external_deadline is not None:
            self.deadline = min(self.deadline, external_deadline)
        self.workspace = DocumentWorkspace(manifest, self.directory / "workspace", identity,
            require_semantic_review=self.settings.require_semantic_review,
            allow_quality_gaps=self.settings.allow_quality_gaps,
            **({"material_index": material_index} if material_index is not None else {}))
        self.source_delivery = None
        if self.settings.single_document_agent:
            from .document_source_delivery import SourceDeliveryLedger
            self.source_delivery = SourceDeliveryLedger(manifest, self.directory / "source_delivery",
                run_id=json_digest(str(self.directory.resolve())), agent_id="document-main",
                on_confirmed=self.workspace.record_read)
        self._tools = None
        self._workspace_tools = None
        self._review_journal = None
        self._tools_type = DocumentTools
        self._tool_configuration = dict(manifest=manifest, count_tokens=self._count_tokens,
            max_tool_tokens=self.settings.tool_tokens, source_path=source_path,
            vision_backend=vision_backend, on_read=self.workspace.record_read,
            on_image_observation=self.workspace.record_image_observation,
            native_response_budget=True, compact_model_source=self.settings.compact_model_source)
        if self.source_delivery is not None:
            self._tool_configuration["delivery_ledger"] = self.source_delivery
        if material_index is not None:
            self._tool_configuration["material_index"] = material_index
        self.tool_calls_before_resume = previous.get("tool_calls", [])
        self.image_observations_before_resume = previous.get("image_observations", [])
        self.current_node = None
        self.review_feedback = self.workspace.snapshot().get("review")
        self.gateway_receipts = previous.get("gateway_receipts", [])
        # Ephemeral native SDK execution pointer: never restored from a past
        # session or supplied by an Agent. Terminal sidecars remain durable.
        self._active_agent_call_id = None
        self._active_review_call_id = None
        self.last_finish = previous.get("accepted_finish")
        self.last_finish_origin = previous.get("actual_finish_origin")
        self.delivery_inspections = copy.deepcopy(previous.get("delivery_inspections", []))
        self.model_self_check = copy.deepcopy(previous.get("model_self_check"))
        self.artifact_delivery = copy.deepcopy(previous.get("artifact_delivery"))
        self._artifact_manager = None
        self.context_state = None
        if self.settings.single_document_agent:
            from .document_context_state import ExactProgressRecorder
            self.context_state = ExactProgressRecorder(self.directory / "exact_context_state", self.identity)
            self.context_state.observe("initial_workspace", {}, {"ok":True}, self.workspace.snapshot())
        self._single_agent_terminal_failure = None
        self.previous_status = previous.get("status")
        self.previous_entry_mode = previous.get("entry_mode", "navigation")
        self.entry_mode = self.previous_entry_mode
        # Controller, independent reviewer and image observations share the
        # mutable client's timeout, cumulative budget and dispatch ledger.
        self._model_forward_lock = threading.Lock()
        if source_catalog is not None:
            from .document_source_objects import DocumentSourceObjects
            self.source_objects = DocumentSourceObjects(
                manifest, Path(source_path), self.directory / "source_objects", identity,
                self._source_terminal_model_bundle, self._count_tokens, self.settings.tool_tokens,
                generate_origin_call_id=self._source_proposal_origin, native_response_budget=True)
            if self.source_objects.catalog["catalog_sha256"] != source_catalog["catalog_sha256"]:
                raise ValueError("Native source catalog changed during workflow initialization")
            self.workspace.source_gate = self.source_objects.bind_state
        self._vision_admission = None
        self._vision_factory = vision_backend_factory
        if vision_backend_factory is not None:
            verify_endpoint = getattr(vision_backend_factory, "verify_endpoint", None)
            if callable(verify_endpoint):
                verify_endpoint(self.client.base_url)
            vision_backend, self._vision_admission = vision_backend_factory(
                generate_observation=self._generate_observation, deadline=self.deadline,
                directory=self.directory / "vision", manifest=self.manifest)
            if not callable(self._vision_admission):
                raise ValueError("Vision factory must return exact native processor admission")
            self._tool_configuration["vision_backend"] = vision_backend
            self._tool_configuration["on_image_observation"] = self.workspace.record_image_observation

    def _generate_observation(self, payload, *, provenance, deadline):
        if self._vision_admission is None or deadline != self.deadline:
            raise ValueError("Image requests must use this report's bound admission and deadline")
        self._check_budget()
        verify_endpoint = getattr(self._vision_factory, "verify_endpoint", None)
        if callable(verify_endpoint):
            verify_endpoint(self.client.base_url)
        admission_identity = {key: self._vision_admission.identity[key]
                              for key in ("processor_identity_sha256", "service_identity_sha256")}
        gateway = AuditedModelGateway(self.client, deadline=self.deadline,
            max_calls=self.settings.max_model_calls, max_output_tokens=self.settings.output_tokens,
            before_request=self._check_budget, request_kind="generation",
            request_admission=self._vision_admission, admission_identity=admission_identity,
            forward_lock=self._model_forward_lock, generation_purpose="image_observation",
            on_dispatch=self._on_dispatch, on_response=self._on_model_response,
            on_prepared_request=self._on_model_request)
        gateway.receipts = self.gateway_receipts
        gateway.on_receipt = self._on_model_receipt
        first = len(self.gateway_receipts)
        response = gateway.forward(payload)
        audit = self.gateway_receipts[first:]
        if len(audit) != 1 or audit[0].get("status") != "success":
            raise ValueError("Image generation lacks a settled gateway receipt")
        audit = copy.deepcopy(audit[0])
        receipts = [call for call in self._model_calls() if call.get("call_id") == audit.get("call_id")]
        if len(receipts) != 1:
            raise ValueError("Image generation lacks a unique actual HTTP call")
        return {"response": response, "receipt": receipts[0], "audit": audit,
                "admission": copy.deepcopy(audit["admission"])}

    @property
    def tools(self):
        # Completed publication retries need no SDK or remote tokenizer. Also
        # keep expired-run failures inside run(), where they receive a receipt.
        if self._tools is None:
            self._tools = self._tools_type(**self._tool_configuration)
            if self.settings.compact_model_source:
                atomic_json(self.directory / "model-source-map.json", self._tools.model_source.source_map())
            self._tools.receipts = self.tool_calls_before_resume
            self._tools.image_observations = self.image_observations_before_resume
        return self._tools

    def _tool_calls(self):
        return self._tools.receipts if self._tools is not None else self.tool_calls_before_resume

    @property
    def workspace_tools(self):
        from .document_workspace_tools import WorkspaceTools
        if self._workspace_tools is None:
            self._workspace_tools = WorkspaceTools(
                self.workspace, count_tokens=self._count_tokens,
                max_tool_tokens=self.settings.tool_tokens, native_response_budget=True)
        return self._workspace_tools

    @property
    def review_journal(self):
        from .document_review_journal import ReviewJournal
        if self._review_journal is None:
            self._review_journal = ReviewJournal(
                self.directory / "semantic_review",
                {"workflow": self.identity, "protocol": REVIEW_VERSION})
        return self._review_journal

    def _image_observations(self):
        return self._tools.image_observations if self._tools is not None else self.image_observations_before_resume

    def _observations_for_image(self, image_id, ref=None):
        ref = ref or {}
        identifiers = ([ref["observation_id"]] if "observation_id" in ref
                       else ref.get("observation_ids"))
        index = self.workspace.snapshot().get("image_observations", {})
        if identifiers is None:
            identifiers = [key for key, value in index.items() if value["image_id"] == image_id]
        observations = [self.workspace.load_image_observation(image_id, identifier)
                        for identifier in identifiers]
        if not observations and not index and not ref.get("observation_id") and not ref.get("observation_ids"):
            # Old OCR checkpoints did not have immutable observation IDs.
            observations = [copy.deepcopy(value) for value in self._image_observations()
                            if value.get("image_id") == image_id]
        return observations

    def _model_calls(self):
        return [*self.model_calls_before_resume, *copy.deepcopy(getattr(self.client, "calls", [])),
                *copy.deepcopy(getattr(self.compression_client, "calls", []))]

    def _check_budget(self, *, include_model=True):
        if self.native_compression is not None and self.native_compression.failure is not None:
            raise RuntimeError('Native compression failed; original memory and saved workspace retained')
        if self.settings.single_document_agent and getattr(self, "_single_agent_terminal_failure", None) is not None:
            raise RuntimeError("Continuous Agent model request failed or truncated; progress saved, no automatic retry")
        if self.settings.dynamic_output_budget and getattr(self.client, "_dynamic_budget_failure", None) is not None:
            raise RuntimeError("Dynamic complete-request admission failed; no report fallback permitted")
        if include_model and any(d.get("status") == "outcome_unknown" for d in self.model_dispatches):
            raise RuntimeError("Prior generation outcome is unknown; reconcile its request receipt before replay")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Report's cumulative time budget exhausted")
        generations = [r for r in self._model_calls() if r.get("kind") in {"generation", "agent_generation"}]
        if include_model and len(generations) >= self.settings.max_model_calls:
            raise RuntimeError("Report's cumulative model request budget exhausted")

    def _timed(self, function, *args, **kwargs):
        self._check_budget(include_model=False)
        original = self.client.timeout
        self.client.timeout = min(original, self.deadline - time.monotonic())
        try:
            result = function(*args, **kwargs)
            self._check_budget(include_model=False)
            return result
        finally:
            self.client.timeout = original

    def _count_tokens(self, text):
        return self._timed(self.client.count_tokens, text)

    def _count_chat_tokens(self, messages):
        return self._timed(self.client.count_chat_tokens, messages)

    def _persist(self, status="running"):
        # A terminal callback may fail after seeing success, whereupon the
        # gateway rejects that delivery. Reconcile its final audit on every
        # later save, including a tool's finally and resumed checkpoints.
        terminal = {receipt["request_id"]: receipt for receipt in self.gateway_receipts
                    if receipt.get("request_id") and receipt.get("status") in {"success", "rejected"}}
        model_calls = self._model_calls()
        for dispatch in self.model_dispatches:
            receipt = terminal.get(dispatch["request_id"])
            if receipt is not None:
                dispatch.update(status=("outcome_unknown" if _generation_outcome_unknown(
                                            receipt, model_calls) else "settled"),
                                    receipt_status=receipt["status"],
                                generation_attempted=receipt.get("generation_attempted", False),
                                call_id=receipt.get("call_id"))
        atomic_json(self.trace_path, {
            "schema_version": 1, "identity": self.identity, "status": status,
            "dag_nodes": self.nodes, "sessions": self.sessions,
            "review_sessions": self.review_sessions, "pending_review": self.pending_review,
            "handoff_events": self.handoff_events,
            "reading_terminal_attempts": self.reading_terminal_attempts,
            "review_attempts": self.review_attempts, "spent_seconds": self.total_spent_seconds + time.monotonic() - self.started,
            "model_calls": model_calls, "gateway_receipts": self.gateway_receipts,
            "model_dispatches": self.model_dispatches,
            "uncertain_budget_seconds": self.uncertain_budget_seconds,
            "tool_calls": self._tool_calls(), "delivery": self.workspace.delivery_status(),
            "image_observations": self._image_observations(),
            "accepted_finish": self.last_finish, "actual_finish_origin": self.last_finish_origin,
            "delivery_inspections": self.delivery_inspections, "model_self_check": self.model_self_check,
            "artifact_delivery": self.artifact_delivery,
            "entry_mode": self.entry_mode,
        })

    def _on_dispatch(self, dispatch):
        if dispatch.get("request_kind") == "agent_generation":
            self._active_agent_call_id = None
        record = {**copy.deepcopy(dispatch), "status": "pending",
                  "remaining_report_seconds": max(0.0, self.deadline - time.monotonic())}
        self.model_dispatches.append(record)
        try:
            self._persist()
        except Exception:
            self.model_dispatches.pop()
            raise

    def _on_model_request(self, payload, dispatch):
        if dispatch.get("request_kind") == "agent_generation":
            self._active_agent_call_id = None
        if json_digest(payload) != dispatch["request_sha256"]:
            raise ValueError("Prepared model request differs from dispatch")
        path = self.directory / "model_requests" / (dispatch["request_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.workspace.snapshot()
        atomic_json(path, {"schema_version": 1,
            "workflow_identity_sha256": json_digest(self.identity),
            "request_sha256": dispatch["request_sha256"], "request_id": dispatch["request_id"],
            "workspace_version": snapshot["version"],
            "semantic_state_sha256": self._semantic_identity(snapshot),
            "request": copy.deepcopy(payload)})
        if getattr(self, "source_delivery", None) is not None:
            self.source_delivery.observe_request(payload, dispatch)

    def _source_input_observed(self, payload, audit):
        """Gateway-proven input delivery, even if output arguments later fail.

        Dispatch, HTTP headers and SDK parsing are not independent delivery
        proofs. The gateway invokes this one callback only after it observes
        a real upstream response or generation, with the final request audit.
        """
        if getattr(self, "source_delivery", None) is None:
            return
        if json_digest(payload) != audit.get("request_sha256"):
            raise ValueError("Source input acknowledgement differs from prepared request")
        self.source_delivery.confirm_request(audit["request_id"], evidence=audit)

    def _on_model_receipt(self, receipt):
        recoverable_loop = (receipt.get("recovery_authorized") is True
            and receipt.get("error_code") in {"content_repetition", "reasoning_repetition", "raw_repetition", "native_repetition", "content_budget_exceeded", "tool_argument_contract"}
            and generation_stopped(receipt))
        recoverable_length = (receipt.get("recovery_authorized") is True
            and receipt.get("error_code") == "output_token_limit"
            and receipt.get("error") == "VerifiedGenerationLengthError"
            and receipt.get("length_usage_verified") is True
            and receipt.get("http_status") == 200 and receipt.get("finish_reason") == "length")
        failed = receipt.get("failed_generation") or {}
        recoverable_tool_protocol = (receipt.get("recovery_authorized") is True
            and receipt.get("error_code") == "incomplete_tool_call"
            and failed.get("terminal_verified_by") == "gateway_exact_request_admission"
            and failed.get("usage_known") is True
            and failed.get("usage") == failed.get("observed_usage")
            and failed.get("request_id") == receipt.get("request_id")
            and failed.get("call_id") == receipt.get("call_id")
            and verified_incomplete_tool_terminal(failed, model=self.client.model,
                prompt_tokens=receipt.get("prompt_tokens_checked"), max_tokens=receipt.get("max_tokens"),
                max_model_len=self.client.max_model_len))
        if self.settings.single_document_agent and receipt.get("request_kind") == "agent_generation" and (
                receipt.get("generation_attempted") is True and receipt.get("status") != "success"
                or receipt.get("finish_reason") == "length") and not (recoverable_loop or recoverable_length or recoverable_tool_protocol):
            self._single_agent_terminal_failure = {"error": receipt.get("error"),
                "status": receipt.get("status"), "finish_reason": receipt.get("finish_reason"),
                "request_id": receipt.get("request_id"), "call_id": receipt.get("call_id")}
        if receipt.get("request_kind") == "agent_generation":
            self._active_agent_call_id = None
        prior = copy.deepcopy(self.model_dispatches)
        for dispatch in self.model_dispatches:
            if dispatch["request_id"] == receipt.get("request_id"):
                dispatch.update(status=("outcome_unknown" if _generation_outcome_unknown(
                                            receipt, self._model_calls()) else "settled"),
                                    receipt_status=receipt["status"],
                                generation_attempted=receipt.get("generation_attempted", False),
                                call_id=receipt.get("call_id"))
                break
        try:
            self._persist()
        except Exception:
            self.model_dispatches[:] = prior
            raise
        if receipt.get("request_kind") == "agent_generation" and receipt.get("status") == "success":
            if self.source_objects is not None:
                self._terminal_model_bundle(receipt["call_id"])
            self._active_agent_call_id = receipt["call_id"]

    def _on_model_response(self, response, audit):
        """Preserve validated raw model replies before SDK argument parsing.

        These local sidecars contain report-derived text, never HTTP headers or
        credentials. Public/summary receipts continue to contain only hashes.
        A malformed native tool argument can then be diagnosed from the exact
        bytes rather than guessed from a finish reason or a shape summary.
        """
        record = {"schema_version": 1, "workflow_identity_sha256": json_digest(self.identity),
                  "request_id": audit["request_id"], "call_id": audit["call_id"],
                  "request_sha256": audit["request_sha256"],
                  "response_sha256": audit["response_sha256"], "response": response}
        if json_digest(response) != record["response_sha256"]:
            raise ValueError("Model response differs from its validated receipt")
        path = self.directory / "model_responses" / (json_digest({"call_id": record["call_id"]}) + ".json")
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("A model response sidecar cannot overwrite another response")
        atomic_json(path, record)

    def _terminal_model_bundle(self, call_id):
        """Join actual terminal audits with their durable complete wire sidecars."""
        matches = [a for a in self.gateway_receipts if a.get("call_id") == call_id]
        if len(matches) != 1:
            raise ValueError("Model call needs a unique terminal gateway audit")
        audit = copy.deepcopy(matches[0])
        calls = [c for c in self._model_calls() if c.get("call_id") == call_id]
        if (audit.get("status") != "success" or len(calls) != 1
                or calls[0].get("status") != "success" or calls[0].get("http_status") != 200
                or calls[0].get("kind") != audit.get("request_kind")
                or calls[0].get("input_sha256") != audit.get("request_sha256")
                or calls[0].get("request_id") != audit.get("request_id")):
            raise ValueError("Model call lacks its actual successful HTTP receipt")
        request_path = self.directory / "model_requests" / (audit["request_id"] + ".json")
        response_path = self.directory / "model_responses" / (json_digest({"call_id": call_id}) + ".json")
        request = json.loads(request_path.read_text())
        response = json.loads(response_path.read_text())
        for record in (request, response):
            if (record.get("workflow_identity_sha256") != json_digest(self.identity)
                    or record.get("request_id") != audit["request_id"]
                    or record.get("request_sha256") != audit["request_sha256"]):
                raise ValueError("Model sidecar workflow/request binding differs")
        if (response.get("call_id") != call_id
                or response.get("response_sha256") != audit["response_sha256"]
                or json_digest(request.get("request")) != audit["request_sha256"]
                or json_digest(response.get("response")) != audit["response_sha256"]):
            raise ValueError("Model sidecar contents differ from the terminal receipt")
        return {"audit": audit, "request": request["request"], "response": response["response"],
                "request_workspace_version": request.get("workspace_version"),
                "request_semantic_state_sha256": request.get("semantic_state_sha256")}

    def _source_terminal_model_bundle(self, call_id):
        """Preserve the source ledger's narrow v1 receipt contract after validation."""
        bundle = self._terminal_model_bundle(call_id)
        if bundle["request"].get("stream") is True:
            from .source_object_dispositions import STREAM_RECEIPT_FIELDS, validate_source_transport
            # _terminal_model_bundle already joined exactly one actual call
            # with the persisted request/response. Preserve its transport facts
            # inside the existing audit without widening the ledger envelope.
            call = next(row for row in self._model_calls() if row.get("call_id") == call_id)
            bundle["audit"]["stream_transport_receipt"] = {
                key: copy.deepcopy(call.get(key)) for key in STREAM_RECEIPT_FIELDS}
            validate_source_transport(bundle["audit"], bundle["request"], bundle["response"])
        return {key: bundle[key] for key in ("audit", "request", "response")}

    def _source_proposal_origin(self, tool_name, arguments):
        # Only structured native function calls are eligible. Do not search a
        # rationale/free-text response for a JSON-looking proposal or take the
        # last call without matching the exact actual tool arguments.
        if self._active_agent_call_id is None:
            raise ValueError("There is no current terminal successful Agent response")
        bundle = self._terminal_model_bundle(self._active_agent_call_id)
        if bundle["audit"].get("request_kind") != "agent_generation":
            raise ValueError("The current source proposal must come from native Agent generation")
        matches = []
        for choice in bundle["response"].get("choices", []):
            message = choice.get("message", {})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {})
                if function.get("name") != tool_name or tool_call.get("type") != "function":
                    continue
                raw = function.get("arguments")
                try:
                    parsed = _decode(raw) if isinstance(raw, str) else None
                except (ValueError, TypeError):
                    continue
                if (tool_name == "inspect_delivery" and isinstance(parsed, dict)
                        and "cursor" not in arguments and "view" in parsed
                        and set(parsed) <= {"view", "limit", "restart"}):
                    # This tool's documented host defaults are the only
                    # normalization. Explicit cursors and all other tools
                    # keep their exact argument binding.
                    parsed = {"limit":4, "restart":False, **parsed}
                    if type(parsed["limit"]) is not int or type(parsed["restart"]) is not bool:
                        continue
                if (tool_name == "finish_report" and self._native_finish_tool_name() == tool_name
                        and isinstance(parsed, dict) and "basis" in parsed
                        and set(parsed) <= {"basis", "notes"}):
                    parsed = {"notes": "", **parsed}
                if (tool_name == "complete_review" and isinstance(parsed, dict)
                        and "basis" in parsed and set(parsed) <= {"basis", "notes", "gap_checks"}):
                    # Match the SDK's declared top-level container decoding,
                    # including JSON-object strings produced by tool parsers.
                    # This is format conversion only; the original response
                    # remains on disk, and evidence text must still match.
                    from .tong_react_adapter import _normalize_container_arguments
                    from .document_main_review_tools import SCHEMAS
                    try:
                        parsed, _, issues = _normalize_container_arguments(parsed, SCHEMAS[tool_name])
                    except (ValueError, TypeError):
                        continue
                    if issues:
                        continue
                    # The SDK function supplies this declared string default.
                    # Model evidence conclusions remain exact, never supplied
                    # by the host when absent from the actual response.
                    parsed = {"notes": "", **parsed}
                if parsed == arguments:
                    matches.append(self._active_agent_call_id)
        if len(matches) != 1:
            raise ValueError("No unique actual Agent function call matches this source-object proposal")
        return matches[0]

    def _source_status(self, snapshot=None):
        if self.source_objects is None:
            return None
        return self.source_objects.bind_state(snapshot if snapshot is not None else self.workspace.snapshot())

    def _source_handler(self, function):
        def call(**arguments):
            self._source_status()
            return function(**arguments)
        return call

    def _generate_source_review(self, messages, output_tokens):
        self._check_budget()
        gateway = AuditedModelGateway(self.client, deadline=self.deadline,
            max_calls=self.settings.review_iterations, max_output_tokens=self.settings.output_tokens,
            before_request=self._check_budget, request_kind="generation",
            forward_lock=self._model_forward_lock, generation_purpose="source_object_review",
            on_dispatch=self._on_dispatch, on_response=self._on_model_response,
            on_prepared_request=self._on_model_request)
        gateway.receipts = self.gateway_receipts
        gateway.on_receipt = self._on_model_receipt
        first = len(self.gateway_receipts)
        source_request = {"model": self.client.model, "messages": copy.deepcopy(messages),
            "stream": False,
            **text_generation_parameters(max_tokens=output_tokens, temperature=self.settings.temperature,
                                         enable_thinking=self.settings.enable_thinking)}
        gateway.forward(gateway.prepare_request(source_request))
        audits = self.gateway_receipts[first:]
        if len(audits) != 1 or audits[0].get("status") != "success":
            raise ValueError("Source-object review needs a unique settled model receipt")
        self._terminal_model_bundle(audits[0]["call_id"])
        return audits[0]["call_id"]

    @staticmethod
    def _finish_binding(delivery):
        result = {"workspace_version": delivery["workspace_version"], "draft_sha256": delivery["draft_sha256"]}
        if delivery.get("semantic_review", {}).get("performed") is False:
            result["review_not_run_by_user_policy"] = True
        source = delivery.get("source_objects")
        if source is not None:
            result.update(source_object_attestation_sha256=source["attestation_sha256"],
                          source_object_semantic_input_sha256=source["semantic_input_sha256"],
                          source_object_processing_scope_sha256=json_digest(source["processing_scope"]))
        return result

    def _node(self, kind: str) -> dict:
        node = {"id": f"N{len(self.nodes) + 1:06d}", "kind": kind,
                "depends_on": [self.nodes[-1]["id"]] if self.nodes else [],
                "workspace_version": self.workspace.snapshot()["version"], "status": "running"}
        self.nodes.append(node)
        return node

    def _guard(self, function):
        def call(**arguments):
            self._check_budget(include_model=False)
            try:
                return function(**arguments)
            finally:
                self._persist()
        return call

    def _observe_native_tool(self, name, arguments, response):
        if self.context_state is not None:
            self.context_state.observe(name, arguments, response, self.workspace.snapshot())
        # Only an actual successful native tool envelope counts as delivered.
        for page in self._artifact_tool_payloads(response):
            self.delivery_inspections.append(copy.deepcopy(page["inspection_proof"]))
            self._persist()

    def _artifact_tool_payloads(self, envelope):
        if (type(envelope) is dict and envelope.get("ok") is True
                and envelope.get("tool") == "inspect_delivery"
                and type(envelope.get("result")) is dict):
            return [envelope["result"]]
        return []

    def _update_workspace(self, update_json: str):
        update = _normalize_working_notes(_decode(update_json))
        if "row_append" in update:
            snapshot = self.workspace.snapshot()
            existing_ids = set(snapshot["facts"]) | set(snapshot["relations"]) | set(snapshot["questions"])
            update = _normalize_row_append(update, existing_ids)
        # DocumentWorkspace owns the same delivery schema and validates the
        # complete atomic update. Its typed failures retain exact repair paths.
        self.workspace.apply_update(update)
        return self.workspace_tools.summary(tool_name="update_workspace")

    @staticmethod
    def _validate_delivery_schema(draft):
        validate_delivery_data(draft)
        _parse_report(_compact(draft))

    def _output_items(self, snapshot):
        draft, bindings = snapshot["draft"], snapshot["draft_bindings"]
        records = {**snapshot["facts"], **snapshot["relations"], **snapshot["questions"]}
        result = []
        entries = [(f"基本信息.{k}", {k: v}) for k, v in draft["基本信息"].items() if v is not None and v != ""]
        entries += [(f"{section}.{i}", row) for section in ("维修建议", "病害明细") for i, row in enumerate(draft[section])]
        for path, value in entries:
            identifiers = list(bindings[path])
            closure = []
            while identifiers:
                identifier = identifiers.pop()
                if identifier in closure:
                    continue
                closure.append(identifier)
                record = records[identifier]
                identifiers.extend(record.get("depends_on", []))
                identifiers.extend(record[k] for k in ("subject_id", "object_id") if record.get(k) in records)
            refs = [ref for identifier in closure for ref in records[identifier].get("source_refs", [])]
            source = []
            for ref in refs:
                evidence_id = ref.get("evidence_id")
                if evidence_id:
                    unit = next(u for u in self.manifest["units"] if u["id"] == evidence_id)
                    start, end = ref.get("start", 0), ref.get("end", len(unit["text"]))
                    source.append({"evidence_id": evidence_id, "start": start, "end": end,
                                   "source_path": unit["source_path"], "text": unit["text"][start:end],
                                   "context": unit.get("context", {}),
                                   "rendered_context": render_spans(self.manifest, [{"coverage_id": evidence_id,
                                                                                "start": start, "end": end}])["text"]})
                elif ref.get("image_id"):
                    observations = self._observations_for_image(ref["image_id"], ref)
                    if not observations:
                        raise ValueError("Image evidence has no preserved observation")
                    source.append({"image_id": ref["image_id"], "observations": observations,
                                   "verification": "Tool observations are uncertain, not original-report facts"})
            result.append({"path": path, "value": value, "facts": [records[i] for i in closure], "sources": source})
        return self.tools.model_source.project(result) if self.settings.compact_model_source else result

    @staticmethod
    def _semantic_identity(snapshot):
        return json_digest({key: snapshot.get(key) for key in (
            "facts", "relations", "questions", "facts_version", "draft", "draft_fact_ids",
            "draft_bindings", "draft_sha256", "draft_valid", "exclusions", "image_observations")})

    def _review_assignment(self, payload, catalog_sha256):
        """Bind a short acknowledgement to the complete immutable review scope.

        The model still sees every assigned source range and the draft lookup.
        An acknowledgement is its declaration of review, not evidence that it
        understood the source. Exact scopes remain in the journal on disk.
        """
        identity = {
            "protocol": REVIEW_VERSION, "workflow_sha256": json_digest(self.identity),
            "semantic_state_sha256": self._semantic_identity(self.workspace.snapshot()),
            "payload_sha256": json_digest(payload), "draft_catalog_sha256": catalog_sha256,
            "output_item_count": len(payload.get("output_items", [])),
            "source_interval_count": len(payload.get("source_spans", [])),
        }
        return {"id": "ra_" + json_digest(identity)[:32], **identity}

    def _project_review_payload(self, payload):
        return self.tools.model_source.project(payload) if self.settings.compact_model_source else payload

    def _prepare_review_request(self, payload):
        """Budget the complete draft view or its actual read-only tool protocol."""
        lookup = None
        tools = None
        prepared = copy.deepcopy(payload)
        if payload.get("direction") in {"FORWARD", "REVERSE"}:
            from .document_review_lookup import FrozenDraftLookup, LOOKUP_TOOLS
            items = getattr(self, "_active_review_items", None)
            if items is None:
                items = self._output_items(self.workspace.snapshot())
            self._frozen_review_catalog(items=items)
            lookup = FrozenDraftLookup(items, count_tokens=self._count_review_tool_envelope,
                max_tool_tokens=self.settings.tool_tokens, read_source=self.tools.read_source,
                source_manifest=self.manifest, compact_model_source=self.settings.compact_model_source)
            prepared["draft_catalog"] = lookup.metadata(full=True)
            prepared["source_catalog"] = lookup.source.metadata()
            tools = LOOKUP_TOOLS
        prepared["review_assignment"] = self._review_assignment(
            payload, lookup.snapshot_sha256 if lookup is not None else json_digest(payload.get("output_items", [])))
        messages = [{"role": "system", "content": REVIEW_PROMPT},
                    {"role": "user", "content": _compact(prepared)}]
        request = {"model": self.client.model, "messages": messages,
                   "stream": False,
                   **text_generation_parameters(max_tokens=self.settings.output_tokens,
                                                temperature=self.settings.temperature,
                                                enable_thinking=self.settings.enable_thinking)}
        if tools:
            request.update(tools=tools, tool_choice="auto")
            admission = AuditedModelGateway(self.client, deadline=self.deadline,
                max_calls=1, max_output_tokens=self.settings.output_tokens, request_kind="generation")
            complete_fits = (self._timed(admission.count_request, request) + self.settings.planning_output_reserve
                             + self.settings.tool_tokens + self.settings.history_reserve_tokens <= self.settings.max_model_len)
            lookup.initial_values_complete = complete_fits
            if not complete_fits:
                prepared["draft_catalog"] = lookup.metadata(full=False)
                request["messages"][1]["content"] = _compact(prepared)
        return request, lookup

    def _frozen_review_catalog(self, *, items=None, snapshot_sha256=None):
        """Program-side frozen closure; its hash is already in the assignment."""
        digest = json_digest(items) if items is not None else snapshot_sha256
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Frozen review catalog needs a SHA256")
        path = self.directory / "semantic_review" / "catalogs" / (digest + ".json")
        expected = {"schema_version": 1, "workflow_sha256": json_digest(self.identity),
                    "snapshot_sha256": digest, "items": copy.deepcopy(items)} if items is not None else None
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if (type(record) is not dict or set(record) != {
                    "schema_version", "workflow_sha256", "snapshot_sha256", "items"}
                    or record["schema_version"] != 1 or record["workflow_sha256"] != json_digest(self.identity)
                    or record["snapshot_sha256"] != digest or json_digest(record["items"]) != digest
                    or expected is not None and record != expected):
                raise ValueError("Frozen review catalog identity/closure differs")
        elif expected is not None:
            atomic_json(path, expected)
            record = expected
        else:
            raise ValueError("Frozen review catalog is missing")
        return copy.deepcopy(record["items"])

    def _review_dependency_scope(self, payload, items, lookup_calls, call_ids):
        """Derive conservative dependencies from actual native lookup calls.

        A full-draft search/page or an unassignable lookup binds the full
        catalog, including bindings and source/fact versions. Otherwise the
        visible catalog values plus exact assigned/looked-up item closures are
        dependencies. This is scheduling metadata, never a factual judgement.
        """
        by_path = {item["path"]: item for item in items}
        paths = {item["path"] for item in payload.get("output_items", [])}
        full = False
        actual_tools = []
        for call_id in call_ids:
            bundle = self._terminal_model_bundle(call_id)
            for choice in bundle["response"].get("choices", []):
                for tool in choice.get("message", {}).get("tool_calls") or []:
                    function = tool.get("function", {})
                    try:
                        arguments = _decode(function.get("arguments"))
                    except (ValueError, TypeError, AttributeError):
                        continue
                    actual_tools.append((function.get("name"), json_digest(arguments), arguments))
        for call in lookup_calls:
            matching = [arguments for name, digest, arguments in actual_tools
                        if name == call.get("name") and digest == call.get("arguments_sha256")]
            if not matching or call.get("ok") is not True:
                full = True
            if call.get("name") in {"review_read_draft_page", "review_search_draft", "invalid_lookup"}:
                full = True
            if call.get("name") == "review_read_draft_item":
                for arguments in matching:
                    if arguments.get("path") not in by_path or arguments.get("view") not in {"facts", "sources"}:
                        full = True
                    else:
                        paths.add(arguments["path"])
        if paths - by_path.keys():
            raise ValueError("Review dependency scope refers to absent draft items")
        return {"policy": "actual-native-lookups-and-visible-catalog-values-v1",
                "source_snapshot_sha256": json_digest(self.manifest),
                "catalog_values_sha256": json_digest([{"path": item["path"], "value": item["value"]} for item in items]),
                "full_catalog_sha256": json_digest(items) if full else None,
                "item_dependencies": {path: json_digest(by_path[path]) for path in sorted(paths)}}

    def _review_lookup_receipt(self, lookup, payload, call_ids):
        return {"snapshot_sha256": lookup.snapshot_sha256, "tool_calls": copy.deepcopy(lookup.calls),
                "dependency_scope": self._review_dependency_scope(payload, lookup.items, lookup.calls, call_ids)}

    def _current_matches_review_dependencies(self, scope, items):
        by_path = {item["path"]: item for item in items}
        return (scope["source_snapshot_sha256"] == json_digest(self.manifest)
            and scope["catalog_values_sha256"] == json_digest([{"path": i["path"], "value": i["value"]} for i in items])
            and (scope["full_catalog_sha256"] is None or scope["full_catalog_sha256"] == json_digest(items))
            and all(path in by_path and json_digest(by_path[path]) == digest
                    for path, digest in scope["item_dependencies"].items()))

    def _review_prompt_tokens(self, messages):
        request, _ = self._prepare_review_request(_decode(messages[-1]["content"]))
        gateway = AuditedModelGateway(self.client, deadline=self.deadline,
            max_calls=1, max_output_tokens=self.settings.output_tokens, request_kind="generation")
        return (self._timed(gateway.count_request, request) + self.settings.tool_tokens
                + self.settings.history_reserve_tokens)

    def _partial_checkpoint(self, payload, catalog_sha256):
        from .document_review_checkpoint import NativeReviewCheckpoint
        identity = {"workflow": self.identity,
                    "semantic_state_hash": self._semantic_identity(self.workspace.snapshot()),
                    "payload_sha256": json_digest(payload), "catalog_sha256": catalog_sha256}
        return NativeReviewCheckpoint(
            self.directory / "semantic_review" / "partial" / (json_digest(identity) + ".json"), identity)

    def _clear_partial_checkpoint(self, payload):
        self._partial_checkpoint(payload, json_digest(self._active_review_items)).clear()

    def _validate_review_decision(self, decision, payload, assignment):
        arrays = ("unsupported_claims", "missing_items", "conflicts", "unresolved_items")
        if type(decision.get("passed")) is not bool or any(not isinstance(decision.get(k), list) for k in arrays):
            raise ValueError("Semantic review must explicitly report all issue categories")
        expected_paths = {item["path"] for item in payload.get("output_items", [])}
        expected_spans = payload.get("source_spans", [])
        compact_ack = any(k in decision for k in ("checked_assignment_id", "checked_all_assigned"))
        legacy_ack = any(k in decision for k in ("checked_output_paths", "checked_evidence_spans"))
        if compact_ack:
            if (decision.get("checked_assignment_id") != assignment["id"]
                    or decision.get("checked_all_assigned") is not True):
                raise ValueError("Review did not confirm the current complete assignment")
        if legacy_ack or not compact_ack:
            checked_paths = decision.get("checked_output_paths")
            checked_spans = decision.get("checked_evidence_spans")
            if (not isinstance(checked_paths, list) or any(not isinstance(p, str) for p in checked_paths)
                    or len(checked_paths) != len(expected_paths) or set(checked_paths) != expected_paths):
                raise ValueError("Review did not account for every assigned output item")
            if (not isinstance(checked_spans, list) or len(checked_spans) != len(expected_spans)
                    or {json_digest(s) for s in checked_spans} != {json_digest(s) for s in expected_spans}):
                raise ValueError("Review did not account for every assigned source interval")
        if decision["passed"] and any(decision[k] for k in arrays):
            raise ValueError("Review cannot pass while issues remain")
        # Keep model-returned fields untouched. Program-owned assigned scopes
        # are explicitly named, so an auditor cannot mistake them for an echo
        # or semantic verification by the model.
        decision["review_assignment"] = assignment
        decision["assignment_confirmation"] = "compact" if compact_ack else "legacy"
        decision["assigned_output_paths"] = [item["path"] for item in payload.get("output_items", [])]
        decision["assigned_evidence_spans"] = copy.deepcopy(expected_spans)
        return decision

    def _count_review_tool_envelope(self, text):
        # Count the actual native wrapper, including its fixed error envelope.
        # All eight names are considered; a token count is not additive and
        # the longest name in characters need not be longest in tokens.
        from .document_review_lookup import LOOKUP_TOOLS
        value = _decode(text)
        return max(self._count_tokens(_compact(wrap_tool_response(t["function"]["name"], value)))
                   for t in LOOKUP_TOOLS)

    def _review_task_from_request(self, request, payload, assignment=None):
        assignment = assignment or self._review_assignment(payload, json_digest(self._active_review_items))
        tasks = []
        for message in request.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if type(content) is not str:
                continue
            try:
                task = _decode(content)
            except (ValueError, TypeError):
                continue
            if (isinstance(task, dict) and task.get("review_assignment") == assignment
                    and all(task.get(key) == value for key, value in payload.items())):
                tasks.append(task)
        if not tasks or len({json_digest(t) for t in tasks}) != 1:
            raise ValueError("Native review request lost its unique frozen task/assignment")
        return tasks[0], assignment

    def _validate_native_review_request(self, request, payload, assignment=None):
        from .document_review_lookup import LOOKUP_TOOLS
        self._review_task_from_request(request, payload, assignment)
        expected = {t["function"]["name"]: t["function"]["parameters"] for t in LOOKUP_TOOLS}
        tools = request.get("tools")
        if type(tools) is not list or len(tools) != len(expected):
            raise ValueError("Native reviewer needs exactly its readonly tool schemas")
        actual = {}
        for tool in tools:
            function = tool.get("function", {}) if type(tool) is dict else {}
            name = function.get("name")
            if tool.get("type") != "function" or name in actual:
                raise ValueError("Invalid or duplicate native review tool")
            actual[name] = function.get("parameters")
        if actual != expected:
            raise ValueError("Native review readonly schemas differ from the frozen protocol")

    def _native_review_origin(self, payload, name, arguments):
        if self._active_review_call_id is None:
            raise ValueError("No current successful native reviewer call")
        bundle = self._terminal_model_bundle(self._active_review_call_id)
        if bundle["audit"].get("request_kind") != "generation":
            raise ValueError("Reviewer tool cannot use a reading call pointer")
        self._validate_native_review_request(bundle["request"], payload)
        matches = []
        for choice in bundle["response"].get("choices", []):
            for tool in choice.get("message", {}).get("tool_calls") or []:
                function = tool.get("function", {})
                try:
                    parsed = _decode(function.get("arguments"))
                except (ValueError, TypeError):
                    continue
                if (tool.get("type") == "function" and type(tool.get("id")) is str and tool["id"].strip()
                        and function.get("name") == name and parsed == arguments):
                    matches.append(tool.get("id"))
        if len(matches) != 1:
            raise ValueError("Reviewer tool must match one actual complete native function call")
        return self._active_review_call_id

    def _require_stage_boundary(self, outcome=None):
        boundary = session_boundary()
        if not all(boundary.values()):
            raise RuntimeError("Native SDK session lock/active session has not cleared")
        if outcome is not None:
            expected = {"iteration_exited": True, "tool_callbacks_drained": True,
                        "active_session_clear": True, "session_lock_released": True}
            if outcome.get("session_boundary") != expected or outcome.get("error"):
                raise RuntimeError("SDK did not establish a clean returned boundary")
        if any(d.get("status") != "settled" for d in self.model_dispatches):
            raise RuntimeError("Unsettled or unknown model request blocks stage handoff")
        if not self._model_forward_lock.acquire(blocking=False):
            raise RuntimeError("An actual model request still holds the shared forward lock")
        self._model_forward_lock.release()
        self._check_budget(include_model=False)

    def _native_final_decision(self, payload, outcome, call_ids, assignment=None):
        if (outcome.get("status") != "completed" or outcome.get("native_final_event") is not True
                or outcome.get("profile") != "reviewer" or not call_ids):
            raise ValueError("Native reviewer has no complete final SDK verdict")
        self._require_stage_boundary(outcome)
        bundles = [self._terminal_model_bundle(call_id) for call_id in call_ids]
        for bundle in bundles:
            if bundle["audit"].get("request_kind") != "generation":
                raise ValueError("Native reviewer verdict cannot borrow a reading call")
            self._validate_native_review_request(bundle["request"], payload, assignment)
        final = bundles[-1]["response"]["choices"][0]
        content = final.get("message", {}).get("content")
        if (final.get("finish_reason") != "stop" or type(content) is not str
                or final.get("message", {}).get("tool_calls")
                or content.strip() != outcome.get("response_text")):
            raise ValueError("Native final event differs from its complete actual HTTP response")
        _, assignment = self._review_task_from_request(bundles[-1]["request"], payload, assignment)
        decision = self._validate_review_decision(_decode(content), payload, assignment)
        decision.update(model_call_id=call_ids[-1], model_call_ids=list(call_ids),
                        review_engine="native_tongagents", native_adapter_version=NATIVE_REVIEW_ADAPTER_VERSION)
        return decision

    def _validated_cached_review(self, payload, cached, *, historical=False):
        """Reconcile the actual original SDK verdict; never rebind it in code."""
        identifiers = cached.get("model_call_ids", [cached["model_call_id"]])
        if not identifiers or identifiers[-1] != cached["model_call_id"] or len(set(identifiers)) != len(identifiers):
            raise ValueError("Cached native review lacks unique actual model calls")
        assignment = (cached["review_assignment"] if historical else
                      self._review_assignment(payload, json_digest(self._active_review_items)))
        if historical and (payload.get("direction") != "FORWARD" or cached.get("passed") is not True):
            raise ValueError("Only a clean unchanged forward dependency closure may be reused")
        for identifier in identifiers:
            bundle = self._terminal_model_bundle(identifier)
            self._validate_native_review_request(bundle["request"], payload, assignment)
        native = [session for session in self.review_sessions
                  if session.get("status") == "returned" and session.get("model_call_ids")
                  and session["model_call_ids"][-1] == identifiers[-1]
                  and session.get("assignment") == assignment]
        if len(native) != 1 or native[0].get("terminal_failure") or native[0].get("continuation_state"):
            raise ValueError("Cached verdict lacks a unique accepted native SDK session")
        validated = self._native_final_decision(payload, native[0]["outcome"], identifiers, assignment)
        if any(cached.get(k) != v for k, v in validated.items()):
            raise ValueError("Cached native verdict differs from its actual final response")
        lookup = cached.get("draft_lookup", {})
        original_items = self._frozen_review_catalog(snapshot_sha256=assignment["draft_catalog_sha256"])
        if lookup.get("snapshot_sha256") != assignment["draft_catalog_sha256"] or type(lookup.get("tool_calls")) is not list:
            raise ValueError("Cached review lost its frozen lookup dependency binding")
        dependency = self._review_dependency_scope(payload, original_items, lookup["tool_calls"], identifiers)
        if lookup.get("dependency_scope") != dependency:
            raise ValueError("Cached review lookup dependency scope differs from actual native tools")
        if historical and not self._current_matches_review_dependencies(dependency, self._active_review_items):
            return None  # A changed dependency needs a new own-model judgement.
        return copy.deepcopy(cached)

    @staticmethod
    def _known_review_budget_pause(outcome, audits, runner_error=None):
        """A known local scheduling limit, never an HTTP retry permission."""
        if (runner_error is not None or outcome.get("status") != "failed"
                or outcome.get("profile") != "reviewer"
                or outcome.get("failure_reason") != "tool_budget_exceeded"
                or outcome.get("error") or outcome.get("output_truncated")
                or outcome.get("sdk_tool_errors") or not audits
                or outcome.get("session_boundary") != {
                    "iteration_exited": True, "tool_callbacks_drained": True,
                    "active_session_clear": True, "session_lock_released": True}
                or any(a.get("status") != "success" or a.get("http_status") != 200
                       or a.get("generation_attempted") is not True or not a.get("call_id")
                       for a in audits)):
            return False
        rejected = [call for call in outcome.get("tool_calls", []) if call.get("status") != "success"]
        # The adapter checks this budget before argument processing and trusted
        # Agent-call association. Its actual receipt has no stage field.
        return bool(rejected) and all(call.get("error", {}).get("code") == "TOOL_CALL_BUDGET"
            and call.get("args") == {"invalid_or_unprocessed": True}
            and call.get("args_sha256") is None and "agent_call_id" not in call for call in rejected)

    def _review_request(self, payload: dict) -> dict:
        # Both text directions use the real SDK adapter. No direct gateway
        # generation/tool loop and no nested SDK invocation exist here.
        self._require_stage_boundary()
        request, lookup = self._prepare_review_request(payload)
        if lookup is None:
            raise ValueError("Native review requires a frozen readonly lookup")
        checkpoint = self._partial_checkpoint(payload, lookup.snapshot_sha256)
        state = checkpoint.load() or {"status": "ready", "session_index": None,
            "lookup_state": lookup.export_state(), "model_call_ids": [],
            "iterations_spent": 0, "tool_calls_spent": 0}
        lookup.restore_state(state["lookup_state"])
        for call_id in state["model_call_ids"]:
            matching = [c for c in self._model_calls() if c.get("call_id") == call_id and c.get("kind") == "generation"]
            if len(matching) != 1:
                raise ValueError("Native checkpoint lost its actual reviewer call")
        if state["status"] == "started":
            raise RuntimeError("Prior native reviewer has no durable returned SDK boundary; no automatic replay")
        if state["status"] == "returned":
            index = state["session_index"]
            if index >= len(self.review_sessions):
                raise ValueError("Native checkpoint lost its SDK session receipt")
            previous = self.review_sessions[index]
            if (previous.get("assignment") != self._review_assignment(payload, lookup.snapshot_sha256)
                    or previous.get("status") != "returned"):
                raise ValueError("Native review session scope/boundary differs")
            related = [session for session in self.review_sessions if session.get("assignment") == previous["assignment"]]
            if (state["iterations_spent"] != sum(session.get("gateway_receipt_count", 0) for session in related)
                    or [call for session in related for call in session.get("model_call_ids", [])] != state["model_call_ids"]):
                raise ValueError("Native checkpoint counters/calls differ from actual SDK sessions")
            paused = previous.get("continuation_state") == "budget_paused"
            if paused:
                audits = [a for a in self.gateway_receipts if a.get("call_id") in previous["model_call_ids"]]
                if (not self.settings.review_batch_items or previous.get("terminal_failure")
                        or not self._known_review_budget_pause(previous["outcome"], audits)):
                    raise ValueError("Native budget pause lost its known settled provenance")
            elif (previous["outcome"].get("status") == "failed"
                    and previous["outcome"].get("failure_reason") != "missing_or_empty_native_final_response"):
                raise RuntimeError("Prior native reviewer failed; reconcile before resuming")
            if not paused and previous["outcome"].get("native_final_event") is True:
                decision = self._native_final_decision(payload, previous["outcome"], state["model_call_ids"])
                if not lookup.initial_values_complete and decision.get("missing_items") and not lookup.all_values_seen:
                    raise ValueError("Native cached reviewer did not read the full frozen draft")
                decision["draft_lookup"] = self._review_lookup_receipt(lookup, payload, state["model_call_ids"])
                return decision
            if previous["outcome"].get("error") or previous.get("terminal_failure"):
                raise RuntimeError("Prior native reviewer failed; reconcile before resuming")
        from .document_review_lookup import LOOKUP_TOOLS
        while state["iterations_spent"] < self.settings.review_iterations:
            self._check_budget()
            self._require_stage_boundary()
            if len(self.sessions) + len(self.review_sessions) >= self.settings.max_sessions:
                raise RuntimeError("Cumulative reading/reviewer SDK session budget exhausted")
            remaining_tools = self.settings.review_iterations - state["tool_calls_spent"]
            if remaining_tools <= 0:
                raise RuntimeError("Cumulative review tool budget exhausted")
            prepared = _decode(request["messages"][1]["content"])
            if state["iterations_spent"]:
                prepared["native_review_continuation"] = {"prior_attempts": state["iterations_spent"],
                    "saved_notes": lookup.notes, "notes_are_evidence": False,
                    "instruction": "Continue the same frozen assignment. Reread necessary original source/draft; no previous final verdict was accepted."}
            prompt = _compact(prepared)
            index = len(self.review_sessions)
            state.update(status="started", session_index=index)
            self.review_sessions.append({"status": "started", "assignment": prepared["review_assignment"],
                "payload_sha256": json_digest(payload), "engine": "native_tongagents", "direction": payload.get("direction")})
            checkpoint.save(state)
            self._persist()
            first_audit = len(self.gateway_receipts)
            self._active_review_call_id = None

            def receipt(audit):
                self._on_model_receipt(audit)
                state["iterations_spent"] += 1
                if audit.get("call_id") is not None:
                    state["model_call_ids"].append(audit["call_id"])
                if audit.get("status") == "success":
                    bundle = self._terminal_model_bundle(audit["call_id"])
                    self._validate_native_review_request(bundle["request"], payload)
                    self._active_review_call_id = audit["call_id"]
                else:
                    self._active_review_call_id = None
                checkpoint.save(state)

            def prepared_request(actual, dispatch):
                # Preserve the exact incoming SDK request even if its schema
                # or task is rejected before any upstream dispatch.
                self._on_model_request(actual, dispatch)
                self._validate_native_review_request(actual, payload)

            def dispatch_started(dispatch):
                self._active_review_call_id = None
                self._on_dispatch(dispatch)

            def lookup_handler(name):
                def call(**arguments):
                    self._check_budget(include_model=False)
                    self._native_review_origin(payload, name, arguments)
                    if state["tool_calls_spent"] >= self.settings.review_iterations:
                        raise RuntimeError("Cumulative review tool budget exhausted")
                    state["tool_calls_spent"] += 1
                    result = lookup.invoke(name, arguments)
                    state["lookup_state"] = lookup.export_state()
                    checkpoint.save(state)
                    self._persist()
                    return result
                return call

            handlers = {t["function"]["name"]: lookup_handler(t["function"]["name"]) for t in LOOKUP_TOOLS}
            runner_error = None
            tools_before = state["tool_calls_spent"]
            gateway = None
            try:
                with AuditedModelGateway(self.client, deadline=self.deadline,
                        max_calls=min(self.settings.review_iterations - state["iterations_spent"],
                            self.settings.max_model_calls - len([c for c in self._model_calls()
                                if c.get("kind") in {"generation", "agent_generation"}])),
                        max_output_tokens=self.settings.output_tokens, before_request=self._check_budget,
                        request_kind="generation", forward_lock=self._model_forward_lock) as gateway:
                    gateway.receipts = self.gateway_receipts
                    gateway.on_dispatch = dispatch_started
                    gateway.on_prepared_request = prepared_request
                    gateway.on_response = self._on_model_response
                    gateway.on_receipt = receipt
                    config = {**self.react_config, "profile": "reviewer", "url": gateway.url,
                        "system_prompt": REVIEW_PROMPT, "enable_thinking": self.settings.enable_thinking,
                        "temperature": self.settings.temperature, "max_tokens": self.settings.output_tokens,
                        "max_iterations": min(self.settings.session_iterations,
                                              self.settings.review_iterations - state["iterations_spent"]),
                        "max_calls": min(self.settings.session_calls, remaining_tools),
                        "current_agent_call_id": lambda: self._active_review_call_id,
                        "max_tool_tokens": self.settings.tool_tokens, "count_tokens": self._count_tokens,
                        "timeout": min(self.react_config.get("timeout", 180), max(0.01, self.deadline - time.monotonic()))}
                    if self.settings.dynamic_output_budget:
                        config.update(dynamic_output_budget=True, prepare_generation_request=gateway.prepare_request)
                    outcome = self.react_runner(prompt, handlers, config)
            except Exception as exc:
                runner_error = exc
                outcome = {"status": "failed", "native_final_event": False, "profile": "reviewer",
                           "error": {"type": type(exc).__name__}, "tool_calls": []}
            finally:
                self._active_review_call_id = None
                state["tool_calls_spent"] = max(state["tool_calls_spent"], tools_before +
                    len(outcome.get("tool_calls", [])) + len(outcome.get("sdk_tool_errors", [])))
                state["lookup_state"] = lookup.export_state()
                state["status"] = "returned"
                audits = self.gateway_receipts[first_audit:]
                failures = [a for a in audits if a.get("status") != "success"]
                self.review_sessions[index].update(status="returned", outcome=outcome,
                    model_call_ids=[a["call_id"] for a in audits if a.get("call_id")],
                    terminal_failure=bool(failures), gateway_receipt_count=len(audits))
                checkpoint.save(state)
                self._persist()
            self._require_stage_boundary()
            # The SDK may surface an audited length as its HTTP400 exception.
            # Only the actual checked terminal audit can recover that type.
            if failures:
                if len(failures) == 1 and audits[-1] is failures[0] and failures[0].get("error") == "VerifiedGenerationLengthError":
                    audit = failures[0]
                    call = next(c for c in self._model_calls() if c.get("call_id") == audit.get("call_id"))
                    terminals = [item for item in gateway.verified_length_attempts
                                 if item["attempt"].get("call_id") == audit["call_id"]]
                    if len(terminals) != 1:
                        raise ValueError("Native rejected length lost its unique inert terminal evidence")
                    error = VerifiedGenerationLengthError(terminals[0]["attempt"], terminals[0]["response"])
                    self._validate_review_length_origin(payload, error.attempt, error.response)
                    raise error
                raise RuntimeError("Native reviewer request failed; no automatic retry")
            if runner_error is not None:
                raise runner_error
            self._require_stage_boundary(outcome)
            if (self.settings.review_batch_items
                    and self._known_review_budget_pause(outcome, audits, runner_error)):
                self.review_sessions[index].update(continuation_state="budget_paused",
                    final_response_accepted=False,
                    continuation_policy="same-frozen-assignment-saved-model-notes-no-output-replay")
                self._persist()
                # Every dispatched HTTP call has succeeded and the native SDK
                # fully returned. Discard its non-completed final as a verdict;
                # only bounded model-authored lookup notes cross the boundary.
                continue
            if (outcome.get("status") == "failed"
                    and outcome.get("failure_reason") != "missing_or_empty_native_final_response"):
                raise RuntimeError("Native reviewer stage failed; no automatic retry")
            if outcome.get("native_final_event") is True:
                decision = self._native_final_decision(payload, outcome, state["model_call_ids"])
                decision["draft_lookup"] = self._review_lookup_receipt(lookup, payload, state["model_call_ids"])
                if (not lookup.initial_values_complete and decision.get("missing_items") and not lookup.all_values_seen):
                    raise ValueError("Native reviewer asserted omission without reading the frozen full draft")
                return decision
            if not audits:
                raise RuntimeError("Native reviewer returned without an actual model attempt")
        raise RuntimeError("Cumulative native review iteration budget exhausted")

    def _validate_review_length_origin(self, payload, attempt, response):
        """Reconcile terminal evidence; never parse a failed assistant message."""
        required = {"call_id", "request_id", "response_id", "request_sha256",
                    "gateway_canonical_response_sha256", "client_http_response_sha256",
                    "usage", "prompt_tokens_checked"}
        if (type(attempt) is not dict or set(attempt) != required
                or type(response) is not dict):
            raise ValueError("Invalid reverse length provenance fields")
        audits = [a for a in self.gateway_receipts if a.get("call_id") == attempt["call_id"]]
        calls = [c for c in self._model_calls() if c.get("call_id") == attempt["call_id"]]
        dispatches = [d for d in self.model_dispatches if d.get("request_id") == attempt["request_id"]]
        if len(audits) != 1 or len(calls) != 1 or len(dispatches) != 1:
            raise ValueError("Reverse length needs unique actual provenance")
        audit, call, dispatch = audits[0], calls[0], dispatches[0]
        usage = attempt["usage"]
        checked = attempt["prompt_tokens_checked"]
        if (audit.get("status") != "rejected" or audit.get("request_kind") != "generation"
                or audit.get("error") != "VerifiedGenerationLengthError" or audit.get("receipt_callback_error")
                or audit.get("generation_attempted") is not True or audit.get("length_usage_verified") is not True
                or audit.get("http_status") != 200 or audit.get("finish_reason") != "length"
                or audit.get("response_id") != attempt["response_id"]
                or audit.get("request_id") != attempt["request_id"]
                or audit.get("request_sha256") != attempt["request_sha256"]
                or audit.get("response_sha256") != attempt["gateway_canonical_response_sha256"]
                or audit.get("usage") != usage or audit.get("prompt_tokens_checked") != checked
                or call.get("status") != "failed" or call.get("kind") != "generation"
                or call.get("http_status") != 200 or call.get("finish_reason") != "length"
                or call.get("response_id") != attempt["response_id"]
                or call.get("request_id") != attempt["request_id"]
                or call.get("input_sha256") != attempt["request_sha256"]
                or call.get("response_sha256") != attempt["client_http_response_sha256"]
                or call.get("usage") != usage or call.get("prompt_tokens_checked") != checked
                or dispatch.get("status") != "settled" or dispatch.get("generation_attempted") is not True
                or dispatch.get("call_id") != attempt["call_id"]
                or dispatch.get("request_sha256") != attempt["request_sha256"]
                or dispatch.get("request_kind") != "generation"
                or json_digest(response) != attempt["gateway_canonical_response_sha256"]
                or response.get("id") != attempt["response_id"]
                or response.get("model") != self.client.model or response.get("usage") != usage):
            raise ValueError("Reverse length actual provenance differs")
        request_record = json.loads((self.directory / "model_requests" / (attempt["request_id"]+".json")).read_text())
        request = request_record.get("request", {})
        task, assignment = self._review_task_from_request(request, payload)
        if (request_record.get("workflow_identity_sha256") != json_digest(self.identity)
                or request_record.get("request_id") != attempt["request_id"]
                or request_record.get("request_sha256") != attempt["request_sha256"]
                or json_digest(request) != attempt["request_sha256"]
                or task.get("review_assignment") != assignment
                or any(task.get(key) != value for key, value in payload.items())
                or request.get("model") != self.client.model
                or request.get("max_tokens") != (self.settings.max_model_len - checked
                    if self.settings.dynamic_output_budget and type(checked) is int else self.settings.output_tokens)
                or request.get("temperature") != self.settings.temperature
                or request.get("chat_template_kwargs") != {"enable_thinking":self.settings.enable_thinking}
                or type(usage) is not dict or type(checked) is not int
                or type(usage.get("completion_tokens")) is not int
                or not 0 <= usage["completion_tokens"] <= request.get("max_tokens", 0)
                or type(usage.get("total_tokens")) is not int
                or usage.get("prompt_tokens") != checked
                or usage["total_tokens"] != checked+usage["completion_tokens"]
                or usage["total_tokens"] > self.settings.max_model_len
                or len(response.get("choices", [])) != 1
                or response["choices"][0].get("finish_reason") != "length"):
            raise ValueError("Reverse length request/usage/assignment binding differs")

    def _review_length_origin(self, payload, *, error=None, origin=None):
        """Keep failed output outside successful model bundles and prompts."""
        if error is not None:
            attempt, response = copy.deepcopy(error.attempt), copy.deepcopy(error.response)
            self._validate_review_length_origin(payload, attempt, response)
            path = self.directory / "semantic_review" / "failed_length" / (json_digest({"call_id":attempt["call_id"]})+".json")
            record = {"schema_version":1, "workflow_sha256":json_digest(self.identity),
                      "accepted":False, "executable":False, "attempt":attempt, "response":response}
            if path.exists() and json.loads(path.read_text()) != record:
                raise ValueError("Failed length sidecar cannot overwrite different output")
            atomic_json(path, record)
            return {"attempt":attempt, "failed_sidecar_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        if type(origin) is not dict or set(origin) != {"attempt", "failed_sidecar_sha256"}:
            raise ValueError("Invalid split-plan failed sidecar binding")
        attempt = origin["attempt"]
        path = self.directory / "semantic_review" / "failed_length" / (json_digest({"call_id":attempt["call_id"]})+".json")
        raw = path.read_bytes();record = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != origin["failed_sidecar_sha256"]
                or record != {"schema_version":1,"workflow_sha256":json_digest(self.identity),
                    "accepted":False,"executable":False,"attempt":attempt,"response":record.get("response")}):
            raise ValueError("Failed length sidecar identity/checksum differs")
        self._validate_review_length_origin(payload, attempt, record["response"])
        return copy.deepcopy(origin)

    def _reverse_split_plan(self, payload, frozen_state, items):
        from .document_review_checkpoint import ReviewSplitPlan
        identity = {"workflow":self.identity, "policy":REVERSE_LENGTH_VERSION,
                    "semantic_state_sha256":frozen_state, "source_sha256":json_digest(self.manifest),
                    "draft_sha256":self.workspace.snapshot()["draft_sha256"],
                    "catalog_sha256":json_digest(items), "parent_payload_sha256":json_digest(payload)}
        return ReviewSplitPlan(self.directory/"semantic_review"/"split_plans"/(json_digest(identity)+".json"),identity)

    def _reverse_length_children(self, payload, items, background, origin):
        from .document_review_payloads import _split, _make_payload
        spans = payload.get("source_spans", [])
        if (payload.get("direction") != "REVERSE"
                or payload != self._project_review_payload(_make_payload(self.manifest,items,background,spans))
                or sum(s["end"]-s["start"] for s in spans) < 2):
            raise ValueError("Only a complete splittable text reverse payload may bisect")
        left,right = _split(spans)
        if not left or not right:
            raise ValueError("Reverse length split produced an empty child")
        children=[]
        for index,part in enumerate((left,right),1):
            child = self._project_review_payload(_make_payload(self.manifest,items,background,part))
            child["review_length_recovery"] = {
                "policy":REVERSE_LENGTH_VERSION, "parent_payload_sha256":json_digest(payload),
                "failed_call_id":origin["attempt"]["call_id"], "child_index":index, "child_count":2,
                "instruction":"上次原任务输出因length不完整，未接受其中内容。本轮只核对当前source_spans；全稿和全源查询仍可用。独立阅读证据并返回本轮完整JSON确认与全部问题；未完成保持未通过。本轮不代替原父任务通过，也不重放任何截断输出。"}
            children.append(child)
        return children

    def _reverse_review_decisions(self, payload, decide, frozen_state, items, background):
        """At most one bisection of an original parent, with durable routing.

        Children call ``decide`` directly: a child's length is an honest failure,
        never recursively split. Successful child journal entries survive a
        later child's failure. No parent verdict is manufactured or journaled.
        """
        plan = self._reverse_split_plan(payload,frozen_state,items)
        previous = plan.load()
        if previous is None:
            try:
                return [decide(payload)]
            except VerifiedGenerationLengthError as error:
                if self._semantic_identity(self.workspace.snapshot()) != frozen_state:
                    raise ValueError("Workspace changed before reverse length recovery")
                origin = self._review_length_origin(payload,error=error)
                children = self._reverse_length_children(payload,items,background,origin)
                plan.save(origin,children)
                self._persist()
        else:
            if self.review_journal.get(frozen_state,payload) is not None:
                raise ValueError("Split parent must not also have a model verdict")
            origin = self._review_length_origin(payload,origin=previous["failed_attempt"])
            children = self._reverse_length_children(payload,items,background,origin)
            if children != previous["children"]:
                raise ValueError("Split children differ from full frozen source/draft closure")
        decisions=[]
        active_batches = getattr(self, "_active_review_batch_plan", None)
        if active_batches is not None:
            active_batches.split(payload, children)
            if self.current_node is not None:
                self.current_node["batch_progress"] = active_batches.progress()
            self._persist()
        for child in children:
            cached=self.review_journal.get(frozen_state,child)
            if cached is not None:
                identifiers=cached.get("model_call_ids",[cached["model_call_id"]])
                if (type(identifiers) is not list or not identifiers
                        or len(set(identifiers))!=len(identifiers) or identifiers[-1]!=cached["model_call_id"]):
                    raise ValueError("Cached reverse child needs its actual successful calls")
                bundles=[self._terminal_model_bundle(identifier) for identifier in identifiers]
                assignment=self._review_assignment(child,json_digest(items))
                if any(self._review_task_from_request(b["request"], child)[1] != assignment for b in bundles):
                    raise ValueError("Cached reverse child request assignment differs")
                final=bundles[-1]["response"]["choices"][0]
                if final.get("finish_reason")!="stop":
                    raise ValueError("Cached reverse child final response is incomplete")
                validated=self._validate_review_decision(_decode(final["message"]["content"]),child,assignment)
                if any(cached.get(k)!=v for k,v in validated.items()):
                    raise ValueError("Cached reverse child verdict differs from its own-model response")
            decisions.append(decide(child))
        return decisions

    def _pack_forward(self, items: list[dict]):
        pending = []
        for item in items:
            if self.settings.review_batch_items and len(pending) >= self.settings.review_batch_items:
                yield {"direction": "FORWARD", "output_items": pending, "source_spans": []}
                pending = []
            candidate = {"direction": "FORWARD", "output_items": [*pending, item], "source_spans": []}
            messages = [{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": _compact(candidate)}]
            if self._review_prompt_tokens(messages) + self.settings.planning_output_reserve > self.settings.max_model_len:
                if not pending:
                    raise ValueError("One draft item and its source references exceed review context; narrow its evidence spans")
                yield {"direction": "FORWARD", "output_items": pending, "source_spans": []}
                pending = [item]
            else:
                pending.append(item)
        if pending:
            yield {"direction": "FORWARD", "output_items": pending, "source_spans": []}

    def _perform_review_draft(self):
        if not self.settings.require_semantic_review:
            raise RuntimeError("Semantic review is disabled by explicit user policy")
        self._check_budget(include_model=False)
        snapshot = self.workspace.snapshot()
        if not snapshot.get("draft_valid") or not snapshot.get("draft"):
            return self.workspace_tools.invalid_draft_response()
        # A partial draft must reach semantic review without inventing basic
        # information. Full delivery validation remains at finish/publication.
        validate_delivery_data(snapshot["draft"])
        frozen_state = self._semantic_identity(snapshot)
        journal = self.review_journal
        self.review_attempts = max(self.review_attempts, journal.attempt_count)
        if journal.active(frozen_state) is None and self.review_attempts >= self.settings.max_review_attempts:
            raise RuntimeError("Semantic review attempt budget exhausted")
        attempt = journal.begin(frozen_state)
        self.review_attempts = max(self.review_attempts, journal.attempt_count)
        node = self._node("D3_bidirectional_semantic_review")
        previous_node = self.current_node
        self.current_node = node
        node["review_attempt_id"] = attempt["attempt_id"]
        self._persist()
        decisions = []
        batch_plan = None

        def historical_forward(payload):
            prior = journal.unchanged_forward(frozen_state, payload) if self.settings.review_batch_items else None
            return self._validated_cached_review(payload, prior["decision"], historical=True) if prior else None

        def decide_one(payload):
            if self._semantic_identity(self.workspace.snapshot()) != frozen_state:
                raise ValueError("Workspace semantic state changed during independent review")
            cached = journal.get(frozen_state, payload)
            if cached is not None:
                cached = self._validated_cached_review(payload, cached)
                node.setdefault("reused_model_call_ids", []).append(cached["model_call_id"])
                self._clear_partial_checkpoint(payload)
                return cached
            cached = historical_forward(payload)
            if cached is not None:
                node.setdefault("dependency_reuse", []).append({
                    "payload_sha256": json_digest(payload),
                    "origin_semantic_state_sha256": cached["review_assignment"]["semantic_state_sha256"],
                    "current_semantic_state_sha256": frozen_state, "model_call_ids": cached["model_call_ids"],
                    "policy": "same-fixed-source-and-exact-forward-output-fact-source-closure",
                    "assignment_rebound": False})
                node.setdefault("reused_model_call_ids", []).append(cached["model_call_id"])
                return cached
            decision = self._review_request(payload)
            if self._semantic_identity(self.workspace.snapshot()) != frozen_state:
                raise ValueError("Workspace semantic state changed during independent review")
            journal.put(frozen_state, payload, decision)
            self._clear_partial_checkpoint(payload)
            self._persist()
            return decision

        def decide(payload):
            planned = batch_plan is not None and any(
                scope["payload_sha256"] == json_digest(payload) for scope in batch_plan.leaf_scopes())
            try:
                decision = decide_one(payload)
                if planned:
                    batch_plan.complete(payload, decision)
                    node["batch_progress"] = batch_plan.progress()
                    self._persist()
                return decision
            except Exception as exc:
                if planned:
                    batch_plan.block(payload, exc)
                    node["batch_progress"] = batch_plan.progress()
                    self._persist()
                raise

        try:
            items = self._output_items(snapshot)
            self._active_review_items = items
            from .document_review_payloads import reverse_payloads
            forward = list(self._pack_forward(items))
            reverse = list(reverse_payloads(self.manifest, items, snapshot["draft"]["基本信息"],
                count_chat_tokens=self._review_prompt_tokens, max_model_len=self.settings.max_model_len,
                output_tokens=self.settings.planning_output_reserve, source_tokens=self.settings.tool_tokens,
                count_tokens=self._count_tokens, review_prompt=REVIEW_PROMPT,
                acknowledgement_mode="compact",
                assignment_id=lambda payload: self._review_assignment(payload, json_digest(items))["id"],
                project_payload=self._project_review_payload))
            remaining_sessions = self.settings.max_sessions - len(self.sessions) - len(self.review_sessions)
            effective_scopes = list(forward)
            for payload in reverse:
                plan = self._reverse_split_plan(payload, frozen_state, items).load()
                if plan is None:
                    effective_scopes.append(payload)
                else:
                    origin = self._review_length_origin(payload, origin=plan["failed_attempt"])
                    children = self._reverse_length_children(payload, items, snapshot["draft"]["基本信息"], origin)
                    if children != plan["children"]:
                        raise ValueError("Split capacity scope differs from its frozen children")
                    effective_scopes.extend(children)
            if self.settings.review_batch_items:
                from .document_review_batches import ReviewBatchPlan, VERSION as batch_version
                batch_identity = {"workflow_sha256": json_digest(self.identity), "policy": batch_version,
                    "semantic_state_sha256": frozen_state, "source_sha256": json_digest(self.manifest),
                    "catalog_sha256": json_digest(items)}
                batch_plan = ReviewBatchPlan(self.directory / "semantic_review" / "batch_plans" /
                    (json_digest(batch_identity) + ".json"), batch_identity, [*forward, *reverse])
                self._active_review_batch_plan = batch_plan
                for payload in reverse:
                    old_plan = self._reverse_split_plan(payload, frozen_state, items).load()
                    if old_plan is not None:
                        origin = self._review_length_origin(payload, origin=old_plan["failed_attempt"])
                        batch_plan.split(payload, self._reverse_length_children(payload, items,
                            snapshot["draft"]["基本信息"], origin))
                node["batch_progress"] = batch_plan.progress()
                self._persist()
            missing_scopes = sum(journal.get(frozen_state, p) is None and not (
                self.settings.review_batch_items and historical_forward(p))
                for p in effective_scopes)
            # One normal reading session is still needed to inspect the result
            # and explicitly finish. One native session per uncached payload is
            # a lower bound, not a promise (length/continuation may need more).
            required_images = {i["id"] for i in self.manifest.get("images", [])} & set(snapshot["reading_progress"]["review_required_evidence_ids"])
            minimum_sessions = missing_scopes + len(required_images) + 1
            node["native_session_capacity"] = {"remaining": remaining_sessions,
                "uncached_text_scopes": missing_scopes, "image_scopes": len(required_images),
                "minimum_with_reader_finish": minimum_sessions, "is_lower_bound": True,
                "scope": "shared reader+reviewer sessions; stricter than parent reading-only limit"}
            if minimum_sessions > remaining_sessions:
                raise RuntimeError("Shared native SDK session capacity is insufficient for this review scope; use an explicitly scoped training diagnostic")
            for payload in forward:
                decisions.append(decide(payload))
            payloads = reverse
            checked_ids = set()
            for payload in payloads:
                spans = payload["source_spans"]
                decisions.extend(self._reverse_review_decisions(payload,decide,frozen_state,items,
                    snapshot["draft"]["基本信息"]))
                for span in spans:
                    self.workspace.record_read(span["evidence_id"], span["start"], span["end"])
                    checked_ids.add(span["evidence_id"])
                self._persist()
            image_ids = {i["id"] for i in self.manifest.get("images", [])}
            required_images = image_ids & set(snapshot["reading_progress"]["review_required_evidence_ids"])
            checked_observation_ids = set()
            for image_id in sorted(required_images):
                observations = self._observations_for_image(image_id)
                if not observations:
                    raise ValueError("Read the image or record why it cannot be verified before semantic review")
                related = [{"path": item["path"], "value": item["value"]} for item in items
                           if any(source.get("image_id") == image_id for source in item["sources"])]
                decisions.append(decide({"direction": "REVERSE", "output_items": [],
                    "related_output": related, "source_spans": [{"image_id": image_id}],
                    "image_observations": observations, "document_background": snapshot["draft"]["基本信息"],
                    "limitations": "Review uncertain OCR/vision observations; this text reviewer cannot independently see pixels."}))
                checked_observation_ids.update(o["observation_id"] for o in observations if o.get("observation_id"))
                checked_ids.add(image_id)
            current = self.workspace.snapshot()
            if self._semantic_identity(current) != frozen_state:
                raise ValueError("Workspace semantic state changed during independent review")
            source_status = self._source_status(current)
            source_call_ids = []
            if self.source_objects is not None:
                from .source_object_review import SourceObjectReview
                object_node = self._node("D3_source_object_independent_review")
                source_input = source_status["semantic_input_sha256"]
                try:
                    coordinator = SourceObjectReview(self.source_objects.ledger, model=self.client.model,
                        max_model_len=self.settings.max_model_len, output_tokens=self.settings.output_tokens,
                        history_reserve_tokens=self.settings.history_reserve_tokens,
                        count_tokens=self._count_chat_tokens, generate_review=self._generate_source_review,
                        dynamic_output_budget=self.settings.dynamic_output_budget)
                    coordinator.review_pending(self.source_objects.pending_proposal_ids())
                    source_status = self._source_status(self.workspace.snapshot())
                    if source_status["semantic_input_sha256"] != source_input:
                        raise ValueError("Source-object review inputs changed during independent review")
                    source_call_ids = list(dict.fromkeys(row["model_call_id"] for row in source_status["coverage"]["objects"] if row.get("model_call_id")))
                    object_node.update(status="completed", passed=source_status["passed"],
                        model_call_ids=source_call_ids, attestation_sha256=source_status["attestation_sha256"])
                except Exception as exc:
                    object_node.update(status="failed", error=type(exc).__name__)
                    raise
            self._check_budget(include_model=False)
            review = {"draft_sha256": current["draft_sha256"], "workspace_version": current["version"],
                      "checked_evidence_ids": sorted(checked_ids),
                      "checked_image_observation_ids": sorted(checked_observation_ids),
                      "model_call_ids": list(dict.fromkeys([i for d in decisions for i in d.get("model_call_ids", [d["model_call_id"]])] + source_call_ids)),
                      "passed": all(d["passed"] for d in decisions) and (source_status is None or source_status["passed"]), "protocol": REVIEW_VERSION}
            for key in ("unsupported_claims", "missing_items", "conflicts", "unresolved_items"):
                review[key] = [issue for d in decisions for issue in d[key]]
            if source_status is not None:
                review.update(source_object_attestation_sha256=source_status["attestation_sha256"],
                              source_object_semantic_input_sha256=source_status["semantic_input_sha256"],
                              source_object_processing_scope_sha256=json_digest(source_status["processing_scope"]))
                if not source_status["passed"]:
                    review["unresolved_items"].append({"reason": "Source-object dispositions require current independent accepted review", "blockers": source_status["blockers"]})
            self.workspace.record_review(review)
            journal.finish(attempt["attempt_id"])
            self.review_feedback = review
            node.update(status="completed", model_call_ids=review["model_call_ids"], passed=review["passed"])
            # Findings remain complete on disk and can be paged back. Returning
            # all prior facts/draft/history would make long-document recovery
            # grow with every update, despite bounded source-reading tools.
            return self.workspace_tools.summary(tool_name="review_draft")
        except Exception as exc:
            node.update(status="failed", error=type(exc).__name__)
            raise
        finally:
            self.current_node = previous_node
            self.__dict__.pop("_active_review_items", None)
            self.__dict__.pop("_active_review_batch_plan", None)
            self._persist()

    def _review_draft(self):
        if not self.settings.require_semantic_review:
            raise RuntimeError("Semantic review is disabled by explicit user policy")
        """Queue only. The current reader remains in charge until real return."""
        self._check_budget(include_model=False)
        snapshot = self.workspace.snapshot()
        if not snapshot.get("draft_valid") or not snapshot.get("draft"):
            return self.workspace_tools.invalid_draft_response()
        validate_delivery_data(snapshot["draft"])
        origin = self._source_proposal_origin("review_draft", {})
        semantic = self._semantic_identity(snapshot)
        if self.pending_review and self.pending_review.get("semantic_state_sha256") == semantic:
            return {"ok": True, "queued": True, "review_id": self.pending_review["id"],
                    "workspace_version": snapshot["version"], "next_action": "End this reading turn normally; independent review begins only after this SDK returns."}
        record = {"schema_version": 1, "workflow_sha256": json_digest(self.identity),
                  "source_sha256": json_digest(self.manifest), "semantic_state_sha256": semantic,
                  "origin_call_id": origin, "reading_session_index": len(self.sessions), "snapshot": snapshot}
        identifier = "queued_" + json_digest(record)
        path = self.directory / "semantic_review" / "queued" / (identifier + ".json")
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("Frozen queued review cannot overwrite another task")
        atomic_json(path, record)
        self.pending_review = {"id": identifier, "record_sha256": json_digest(record),
                               "semantic_state_sha256": semantic, "origin_call_id": origin,
                               "reading_session_index": len(self.sessions)}
        self._persist()
        return {"ok": True, "queued": True, "review_id": identifier,
                "workspace_version": snapshot["version"], "next_action": "End this reading turn normally; do not claim delivery or use SDK stop. Later changes invalidate this task."}

    def _drain_review_queue(self):
        if not self.settings.require_semantic_review:
            if self.pending_review is not None:
                raise RuntimeError("Disabled review queue cannot be consumed")
            return
        if self.pending_review is None:
            return
        pending = copy.deepcopy(self.pending_review)
        if type(pending) is not dict or set(pending) != {"id", "record_sha256", "semantic_state_sha256", "origin_call_id", "reading_session_index"}:
            raise ValueError("Invalid queued review metadata")
        identifier = pending["id"]
        if type(identifier) is not str or not identifier.startswith("queued_") or len(identifier) != 71 or any(c not in "0123456789abcdef" for c in identifier[7:]):
            raise ValueError("Invalid queued review identifier")
        record = json.loads((self.directory / "semantic_review" / "queued" / (identifier + ".json")).read_text())
        if (json_digest(record) != pending["record_sha256"] or identifier != "queued_" + json_digest(record)
                or record.get("workflow_sha256") != json_digest(self.identity)
                or record.get("source_sha256") != json_digest(self.manifest)
                or record.get("source_sha256") != self.identity["source"]
                or record.get("semantic_state_sha256") != self._semantic_identity(record.get("snapshot", {}))
                or record.get("origin_call_id") != pending["origin_call_id"]
                or record.get("reading_session_index") != pending["reading_session_index"]):
            raise ValueError("Queued review source/snapshot/identity differs")
        index = pending["reading_session_index"]
        if type(index) is not int or not 0 <= index < len(self.sessions):
            raise RuntimeError("Queued reader has not durably returned; no nested reviewer")
        outcome = self.sessions[index]
        self._require_stage_boundary(outcome)
        if pending["origin_call_id"] not in outcome.get("reading_model_call_ids", []):
            raise ValueError("Queued review lost its actual originating reading session")
        bundle = self._terminal_model_bundle(pending["origin_call_id"])
        if bundle["audit"].get("request_kind") != "agent_generation":
            raise ValueError("Queued review origin is not a native reading model call")
        if self._semantic_identity(self.workspace.snapshot()) != pending["semantic_state_sha256"]:
            self.handoff_events.append({"review_id": identifier, "status": "stale_discarded",
                "reason": "Reader changed the semantic workspace after queueing; no model review performed."})
            self.pending_review = None
            self._persist()
            return
        self._perform_review_draft()
        self.handoff_events.append({"review_id": identifier, "status": "review_completed",
                                    "passed": self.workspace.snapshot().get("review", {}).get("passed")})
        self.pending_review = None
        self._persist()

    @property
    def artifact_manager(self):
        if self._artifact_manager is None:
            from .document_delivery_artifacts import DocumentDeliveryArtifacts
            self._artifact_manager = DocumentDeliveryArtifacts(self.directory / "delivery_artifacts",
                json_digest(self.identity), {"workflow_source_sha256":self.identity["source"],
                    "stage":"provided_parsed_text_and_tables", "physical_report_complete":False,
                    **({"submission_format": True} if self.identity.get("run_identity", {}).get("submission_format") else {})})
        return self._artifact_manager

    def _inspect_delivery(self, view, cursor, limit):
        """Legacy explicit-cursor entry; retain its exact proposal binding."""
        call_id = self._source_proposal_origin("inspect_delivery", {"view":view,"cursor":cursor,"limit":limit})
        return self._inspect_delivery_page(view, cursor, limit, call_id)

    def _inspect_delivery_managed(self, view, limit=4, restart=False):
        """Continue this workflow's unchanged actual Word and requested view."""
        if type(restart) is not bool:
            raise ValueError("inspect_delivery restart must be boolean")
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("inspect_delivery limit must be an integer between 1 and 128")
        call_id = self._source_proposal_origin("inspect_delivery", {"view":view,"limit":limit,"restart":restart})
        snapshot = self.workspace.snapshot()
        record = self.artifact_delivery or {}
        cursor = 0
        if not restart and self.artifact_manager.matches_snapshot(snapshot, record):
            pages = [p for p in self.delivery_inspections
                if artifact_page_matches(p, record) and p["view"] == view]
            restarts = [index for index, page in enumerate(pages) if page.get("managed_restart") is True]
            if restarts:
                pages = pages[restarts[-1]:]
            for page in sorted(pages, key=lambda p: (p["cursor"], p["end"])):
                if page["cursor"] > cursor:
                    break
                cursor = max(cursor, page["end"])
        return self._inspect_delivery_page(view, cursor, limit, call_id, managed_restart=restart)

    def _inspect_delivery_page(self, view, cursor, limit, call_id, *, managed_restart=None):
        snapshot = self.workspace.snapshot()
        if (not self.artifact_manager.matches_snapshot(snapshot, self.artifact_delivery)
                or self.artifact_delivery.get("requested_failure_marker")):
            self.artifact_delivery = self.artifact_manager.export(snapshot, incomplete=False,
                structure_status={"phase":"pending_agent_self_check", "complete":False,
                                  "delivery":self.workspace.delivery_status()})
            self.model_self_check = None
        audit = self._terminal_model_bundle(call_id)["audit"]
        size = limit * 512
        while True:
            result = self.artifact_manager.read_page(snapshot, section=view, cursor=cursor, max_chars=size,
                expected_file_sha256=self.artifact_delivery["file_sha256"], receipt=self.artifact_delivery)
            result["inspection_proof"] = {"workspace_version":self.artifact_delivery["workspace_version"],
                "read_workspace_version":snapshot["version"],
                "draft_sha256":snapshot["draft_sha256"],"file_sha256":self.artifact_delivery["file_sha256"],
                "view":view,"cursor":result["cursor"],"end":result["end"],"total_chars":result["total_chars"],
                "call_id":call_id,"request_id":audit["request_id"],
                "request_sha256":audit["request_sha256"],"response_sha256":audit["response_sha256"],
                "tool_result_sha256":json_digest(result)}
            if "content_identity_sha256" in self.artifact_delivery:
                result["inspection_proof"]["content_identity_sha256"] = self.artifact_delivery["content_identity_sha256"]
            if managed_restart is not None:
                result["inspection_proof"]["managed_restart"] = managed_restart
            result["inspection_proof"]["receipt_sha256"] = json_digest(result["inspection_proof"])
            if self._count_tokens(_compact(wrap_tool_response("inspect_delivery",result))) <= self.settings.tool_tokens:
                return result
            if size <= 1:
                raise ValueError("Actual Word metadata cannot fit tool budget; delivery remains incomplete")
            size = max(1, size // 2)

    def _artifact_read_coverage(self, snapshot, pages=None):
        record = self.artifact_delivery or {}
        if (not self.artifact_manager.matches_snapshot(snapshot, record)
                or record.get("requested_failure_marker") is True
                or record.get("structure_valid") is not True or record.get("content_validated") is not True):
            return False
        available_pages = self.delivery_inspections if pages is None else pages
        for view in ("basic","advice","defects"):
            view_pages = [p for p in available_pages if p["view"] == view
                and artifact_page_matches(p, record)]
            if not view_pages or len({p["total_chars"] for p in view_pages}) != 1:
                return False
            end = 0
            for page in sorted(view_pages,key=lambda p:(p["cursor"],p["end"])):
                if page["cursor"] > end:
                    return False
                end = max(end,page["end"])
            if end != view_pages[0]["total_chars"]:
                return False
        # Reopen and verify actual bytes, not a cached model declaration.
        self.artifact_manager.read_page(snapshot, section="summary", cursor=0,max_chars=1,
            expected_file_sha256=record["file_sha256"],receipt=record)
        return True

    def _received_artifact_pages(self, request, snapshot, call_id):
        """Match full actual tool results, not hashes copied into a summary.

        Proofs are accepted only from successful native tool receipts saved by
        this workflow. The current finish request must contain the complete
        page body as a tool message. A request from before an export changed
        cannot be silently attached to the newer file.
        """
        valid_pages = {p["receipt_sha256"]: p for p in self.delivery_inspections
            if self.artifact_manager.matches_snapshot(snapshot, self.artifact_delivery)
            and artifact_page_matches(p, self.artifact_delivery) and p["call_id"] != call_id}
        received, payloads = {}, []
        for message in request.get("messages", []):
            if message.get("role") != "tool" or type(message.get("content")) is not str:
                continue
            try:
                envelope = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            payloads.extend(self._artifact_tool_payloads(envelope))
        for payload in payloads:
            proof = payload.get("inspection_proof")
            if type(proof) is not dict:
                continue
            stored = valid_pages.get(proof.get("receipt_sha256"))
            if stored is None or proof != stored:
                continue
            content = {key: value for key, value in payload.items() if key != "inspection_proof"}
            if stored.get("tool_result_sha256"):
                if json_digest(content) != stored["tool_result_sha256"]:
                    continue
            else:
                # Historical callers have the original proof without a body
                # hash. Validate their complete body against actual Word bytes
                # rather than accepting the old hash-substring shortcut.
                expected = self.artifact_manager.read_page(snapshot, section=stored["view"],
                    cursor=stored["cursor"], max_chars=max(1, stored["end"] - stored["cursor"]),
                    expected_file_sha256=stored["file_sha256"], receipt=self.artifact_delivery)
                if content != expected:
                    continue
            received[stored["receipt_sha256"]] = stored
        return list(received.values())

    def _native_finish_tool_name(self):
        return "finish"

    def _finish_statement_origin(self, self_check):
        return self._source_proposal_origin("finish", {"self_check": self_check})

    def _finish(self, self_check=None):
        result = self.workspace.delivery_status()
        if not result["ready"]:
            from .document_merge_progress import next_blocker_action
            progress = next_blocker_action(self.workspace.snapshot(), result) or {}
            return {"ok": False, "error": "Report cannot be delivered yet",
                    "ready": False, "workspace_version": result["workspace_version"],
                    "draft_sha256": result["draft_sha256"], "blocker_count": len(result["blockers"]),
                    **progress}
        self._validate_delivery_schema(self.workspace.snapshot()["draft"])
        if self.settings.single_document_agent:
            snapshot = self.workspace.snapshot()
            fields = {"inspected_export","complete","basis","remaining_issues"}
            binding_fields = {"workspace_version","draft_sha256","file_sha256"}
            if (type(self_check) is not dict or not fields <= set(self_check)
                    or not set(self_check) <= fields | binding_fields
                    or ("workspace_version" in self_check and type(self_check["workspace_version"]) is not int)
                    or any(type(self_check[key]) is not str for key in ("draft_sha256", "file_sha256") if key in self_check)
                    or type(self_check["inspected_export"]) is not bool or type(self_check["complete"]) is not bool
                    or type(self_check["basis"]) is not str or not self_check["basis"].strip()
                    or type(self_check["remaining_issues"]) is not list
                    or any(type(item) is not str or not item.strip() for item in self_check["remaining_issues"])):
                raise ValueError("Single Agent finish needs its actual file self_check object")
            call_id = self._finish_statement_origin(self_check)
            bundle = self._terminal_model_bundle(call_id)
            semantic = self._semantic_identity(snapshot)
            if bundle.get("request_semantic_state_sha256") != semantic:
                raise ValueError("Self-check request predates the current facts, source bindings or coverage decisions; submit a new model self-check")
            if not self._artifact_read_coverage(snapshot):
                return {"ok":False,"error":"Read all current actual Word basic/advice/defects pages before finish",
                    "next_action":"inspect_delivery","workspace_version":snapshot["version"]}
            request = bundle["request"]
            received_pages = self._received_artifact_pages(request, snapshot, call_id)
            if not self._artifact_read_coverage(snapshot, received_pages):
                raise ValueError("Finish model request has not actually received all current Word pages")
            binding = {"workspace_version":snapshot["version"], "draft_sha256":snapshot["draft_sha256"],
                       "file_sha256":self.artifact_delivery["file_sha256"]}
            for field, value in binding.items():
                if field in self_check and self_check[field] != value:
                    raise ValueError(f"Explicit self_check.{field} does not match the current actual Word; reread the current file")
            self.model_self_check = {"schema":"same-agent-actual-export-self-check-v2", "call_id":call_id,
                "model_statement":copy.deepcopy(self_check), "verified_file_binding":binding,
                "semantic_state_sha256":semantic,
                "content_identity_sha256":self.artifact_delivery.get("content_identity_sha256"),
                "export_workspace_version":self.artifact_delivery["workspace_version"],
                "binding_source":"complete_actual_tool_receipts_in_finish_request",
                "host_semantic_verdict":None,"independent_review":False,
                "actual_inspection_receipt_sha256s":[p["receipt_sha256"] for p in received_pages]}
            if self.settings.allow_quality_gaps:
                self.model_self_check.update(quality_policy="speed_v1_explicit_gaps",
                    exhaustive_quality_verified=False, completion_means="quick_processing_and_actual_file_inspection",
                    unread_source_ranges=self.workspace.snapshot()["reading_progress"]["unread_ranges"])
            if (self_check["inspected_export"] is not True or self_check["complete"] is not True
                    or (self_check["remaining_issues"] and not self.settings.allow_quality_gaps)):
                return {"ok":False,"error":"Agent reports incomplete work; save corrections before finishing",
                    "complete":False,"remaining_issues":copy.deepcopy(self_check["remaining_issues"])}
            audit = self._terminal_model_bundle(call_id)["audit"]
            self.last_finish_origin = {"call_id": call_id, "request_id": audit["request_id"],
                "request_sha256": audit["request_sha256"], "response_sha256": audit["response_sha256"]}
        self.last_finish = self._finish_binding(result)
        if self.settings.single_document_agent:
            self.last_finish.update(file_sha256=self.artifact_delivery["file_sha256"],
                                    content_identity_sha256=self.artifact_delivery.get("content_identity_sha256"),
                                    semantic_state_sha256=self._semantic_identity(self.workspace.snapshot()),
                                    model_self_check_sha256=json_digest(self.model_self_check))
        return {"ok": True, "ready": True, **self.last_finish}

    def _entry(self):
        full = self.tools.full_text()
        prompt = "阅读当前报告，建立工作区并完成任务。\n" + full + "\n图片清单（仍需按需核对）：\n" + _compact(self.manifest.get("images", []))
        messages = [{"role": "system", "content": self._reader_system_prompt()}, {"role": "user", "content": prompt}]
        full_tokens = self._count_chat_tokens(messages)
        # This is only conservative admission. The gateway counts the actual
        # native SDK messages AND tools before issuing any generation request.
        fits = (full_tokens + self.settings.planning_output_reserve
                + (0 if self.settings.dynamic_output_budget else self.settings.history_reserve_tokens + self.settings.tool_tokens)
                <= self.settings.max_model_len)
        if not fits and self.settings.dynamic_output_budget:
            raise ValueError("Full report input occupies context; no navigation fallback permitted")
        if fits:
            return "full", prompt
        return "navigation", self._navigation_prompt()

    def _workspace_bootstrap_prompt(self, user_prompt, session_index):
        """Deliver existing state as a host event, never an SDK tool or proof.

        One existing summary is obtained. Only this added payload can fall back
        to explicit navigation metadata; the original task/source is untouched.
        """
        if not self.settings.workspace_bootstrap:
            return user_prompt
        self._check_budget()
        path = self.directory / "workspace_bootstrap" / f"session-{session_index + 1:03d}.json"
        if path.exists():
            raise ValueError("Workspace bootstrap entry already recorded; reconcile before replay")
        started = time.monotonic()
        record = {"schema": "host-workspace-bootstrap-v2", "status": "preparing",
                  "session_index": session_index, "host_event": True,
                  "native_sdk_tool_event": False, "extra_generation_requests": 0,
                  "notes_are_evidence": False, "read_is_understood": False,
                  "semantic_completion_claim": False,
                  "policy": "current-workspace-entry-snapshot-v2",
                  "dynamic_output_budget": self.settings.dynamic_output_budget,
                  "planning_output_reserve": self.settings.planning_output_reserve,
                  "admission_scope": "system_user_planning_only_gateway_recounts_actual_wire",
                  "complete_chat_is_final_wire_count": False,
                  "dynamic_admission_poisoned": False,
                  "admission_attempts": [],
                  "source_task_sha256": json_digest(user_prompt)}
        instruction = (
            "程序在本次新阅读会话入口提供当前工作区状态；这不是SDK工具调用或完成证明。"
            "已保存的事实、草稿、问题和笔记用于接续工作；笔记只是导航，不是原文证据。"
            + ("已登记阅读范围不代表已理解。工程结论仍要由你依据原文完整上下文判断；本轮不执行语义复核。"
               if self.settings.single_document_agent else
               "已登记阅读范围不代表已理解。工程结论仍要按需回读原文并接受独立语义审核。") +
            "下列版本仅是此刻状态；后续阅读可能推进版本，写入前仍须核对当前version。\n")
        try:
            summary = self.workspace_tools.summary()
            record.update(summary=copy.deepcopy(summary), summary_sha256=json_digest(summary))
            current = self.workspace.snapshot()
            version = summary.get("version", current["version"])
            if version != current["version"]:
                raise ValueError("Workspace changed during bootstrap summary")
            record["workspace_version"] = version

            def assemble(payload):
                added = instruction + "<host_workspace_bootstrap>\n" + _compact(payload) + "\n</host_workspace_bootstrap>\n"
                # Dynamic state follows the stable SYSTEM instruction. Source and
                # dynamic task retain every character supplied by the workflow.
                prompt = added + user_prompt
                tokens = self._count_tokens(added)
                chat_tokens = self._count_chat_tokens([
                    {"role": "system", "content": self._reader_system_prompt()},
                    {"role": "user", "content": prompt}])
                # A dynamic output ceiling equals context; it is not a
                # fixed generation reserve. The exact final SDK tools/history
                # request still receives its own fresh count in the gateway.
                context_fits = (chat_tokens + self.settings.planning_output_reserve
                                + (0 if self.settings.dynamic_output_budget else self.settings.tool_tokens)
                                <= self.settings.max_model_len)
                fits = tokens <= self.settings.history_reserve_tokens and context_fits
                record["admission_attempts"].append({
                    "payload_sha256": json_digest(payload), "added_tokens": tokens,
                    "system_user_tokens": chat_tokens, "context_fits": context_fits,
                    "host_state_budget_fits": tokens <= self.settings.history_reserve_tokens,
                    "planning_output_reserve": self.settings.planning_output_reserve})
                if self.settings.dynamic_output_budget and not context_fits:
                    # Never hide an occupied complete input by replacing its
                    # state with smaller metadata or starting a new session.
                    self.client._dynamic_budget_failure = "WorkspaceBootstrapContextExceeded"
                    record.update(dynamic_admission_poisoned=True,
                                  refusal_stage="host_complete_input_context")
                    raise ValueError("Complete bootstrap input occupies context; no navigation fallback permitted")
                return prompt, tokens, chat_tokens, fits

            prompt, tokens, chat_tokens, fits = assemble(summary)
            payload = summary
            mode = "existing_summary"
            if not fits or summary.get("ok") is not True:
                # Pure indices into existing state. No values, snippets, source
                # ranges or notes are cut; complete records remain in the tools.
                views = ("facts", "relations", "questions", "draft_basic", "draft_advice",
                         "draft_defects", "working_notes", "unread_ranges",
                         "review_required_evidence_ids", "blockers")
                payload = {
                    "ok": True, "mode": "navigation_only", "version": version,
                    "full_summary_sha256": record["summary_sha256"],
                    "full_summary_not_in_prompt": True, "no_record_was_truncated": True,
                    "fallback_reason": ("added_state_budget" if self.settings.dynamic_output_budget else "entry_context_or_added_state_budget") if not fits else "summary_error",
                    "working_notes_are_evidence": False, "read_is_understood": False,
                    "pagination": {"tool": "read_workspace_page", "views": list(views),
                                   "restart": True, "limit": 20,
                                   "consistency": "Host tracks positions separately; changed versions restart the view"},
                    "refresh": {"tool": "read_workspace"},
                }
                prompt, tokens, chat_tokens, fits = assemble(payload)
                mode = "navigation_only"
            record.update(mode=mode, payload=copy.deepcopy(payload), payload_sha256=json_digest(payload),
                          added_tokens=tokens, complete_chat_tokens=chat_tokens,
                          original_task_preserved=True, fits=fits,
                          admitted_user_prompt_sha256=json_digest(prompt),
                          admitted_system_user_sha256=json_digest([
                              {"role": "system", "content": self._reader_system_prompt()},
                              {"role": "user", "content": prompt}]))
            if not fits:
                raise ValueError("Workspace bootstrap navigation metadata cannot fit; no source/state truncated")
            if self.workspace.snapshot()["version"] != version:
                raise ValueError("Workspace changed during bootstrap admission")
            self._check_budget()
            record["status"] = "host_supplied"
            return prompt
        except Exception as exc:
            record.update(status="refused", error_type=type(exc).__name__)
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            atomic_json(path, record)


    def _navigation_prompt(self):
        return "通过 read_outline 导航，再连续阅读原文和表格。无目录则读取 root 章节；重复调用同一工具和对象自动续读，长单元及上下文由程序记录和恢复位置。\n" + _compact({
            "unit_count": len(self.manifest["units"]), "image_count": len(self.manifest.get("images", [])),
            "workspace_version": self.workspace.snapshot()["version"]})

    def _pending_reading_update(self):
        """Locate the latest native write in this run, never replay its bytes."""
        unconfirmed = None
        for session in reversed(self.sessions):
            for call_id in reversed(session.get("reading_model_call_ids", [])):
                bundle = self._terminal_model_bundle(call_id)
                scan = scan_workspace_retry(bundle, session)
                candidate = scan["candidate"]
                if candidate is not None:
                    if candidate["failure_association"].startswith("confirmed_"):
                        return candidate
                    if unconfirmed is None:
                        unconfirmed = candidate
                if scan["supersedes_previous"]:
                    return unconfirmed
        return unconfirmed

    def _verified_reading_length(self, audit, session_index):
        call_id = audit.get("call_id")
        matches = [row for row in self.gateway_receipts if row.get("call_id") == call_id
                   or row.get("request_id") == audit.get("request_id")]
        if len(matches) != 1:
            raise ValueError("Reading length needs one terminal gateway receipt")
        request_path = self.directory / "model_requests" / (audit["request_id"] + ".json")
        request = json.loads(request_path.read_text())
        return reading_length_attempt(audit, self._model_calls(), self.model_dispatches,
                                      request, json_digest(self.identity), session_index)

    def _capture_reading_terminal_attempts(self, audits, session_index):
        """Length is terminal metadata, never a successful raw model bundle."""
        captured = []
        for audit in audits:
            if not (audit.get("request_kind") == "agent_generation"
                    and audit.get("status") == "rejected"
                    and audit.get("generation_attempted") is True
                    and audit.get("http_status") == 200 and audit.get("finish_reason") == "length"):
                continue
            record = self._verified_reading_length(audit, session_index)
            previous = [r for r in self.reading_terminal_attempts if r.get("call_id") == record["call_id"]]
            if previous and previous != [record]:
                raise ValueError("Reading length metadata cannot change or duplicate an attempt")
            if not previous:
                self.reading_terminal_attempts.append(record)
            captured.append(copy.deepcopy(record))
        return captured

    def _reading_length_recovery(self):
        # Only the latest actual Agent terminal may select this strategy. A
        # later success or unknown/ordinary failure never inherits an old length.
        audits = [a for a in self.gateway_receipts if a.get("request_kind") == "agent_generation"]
        if not audits:
            return None
        latest = audits[-1]
        records = [r for r in self.reading_terminal_attempts if r.get("call_id") == latest.get("call_id")]
        if not records:
            return None
        if len(records) != 1 or records[0] != self._verified_reading_length(latest, records[0]["session_index"]):
            raise ValueError("Reading length recovery metadata differs from its actual attempt")
        return {"attempt": copy.deepcopy(records[0]), "instruction": LENGTH_RECOVERY_INSTRUCTION}

    def _post_review_delivery_recovery(self, state, previous):
        # This is feedback for an actually completed handoff, never permission
        # to accept an unreviewed draft or a substitute for the finish tool.
        event = self.handoff_events[-1] if self.handoff_events else {}
        review = state.get("review") or {}
        if (previous.get("status") != "completed"
                or previous.get("native_final_event") is not True
                or previous.get("failure_reason") is not None
                or event.get("status") != "review_completed" or event.get("passed") is not True
                or review.get("passed") is not True or state.get("draft_valid") is not True):
            return None
        delivery = self.workspace.delivery_status()
        if (delivery.get("ready") is not True or delivery.get("blockers") != []
                or delivery.get("workspace_version") != state.get("version")
                or review.get("recorded_version") != state.get("version")
                or delivery.get("facts_version") != state.get("facts_version")
                or review.get("facts_version") != state.get("facts_version")
                or delivery.get("draft_sha256") != state.get("draft_sha256")
                or review.get("draft_sha256") != state.get("draft_sha256")):
            return None
        return {
            "policy": POST_REVIEW_RECOVERY_VERSION,
            "stage": "reviewed_draft_ready_for_delivery",
            "review_id": event["review_id"],
            "workspace_version": state["version"],
            "reviewed_workspace_version": review["workspace_version"],
            "recorded_review_version": review["recorded_version"],
            "facts_version": state["facts_version"],
            "draft_sha256": state["draft_sha256"],
            "review_passed": True, "delivery_ready": True, "blockers": [],
            "next_read": {"tool": "read_workspace"},
            "instruction": POST_REVIEW_RECOVERY_INSTRUCTION,
        }

    def _reading_recovery(self, state, handlers):
        previous = self.sessions[-1] if self.sessions else {}
        draft = state.get("draft") or {}
        post_review = self._post_review_delivery_recovery(state, previous)
        if post_review is not None:
            # Only a current, ready, passed handoff makes earlier errors
            # resolved. Ordinary error and stale-review recovery below is
            # deliberately unchanged, including pending-update provenance.
            recovery = {
                "previous_session_failure": None,
                "previous_sdk_tool_errors": [],
                "previous_handler_tool_errors": [],
                "draft_counts": {"basic": len(draft.get("基本信息", {})),
                                 "advice": len(draft.get("维修建议", [])),
                                 "defects": len(draft.get("病害明细", []))},
                "working_notes": copy.deepcopy(state.get("working_notes", {})),
                "working_notes_are_evidence": False,
                "pending_workspace_retry": None,
                "reading_length_recovery": None,
                "native_review_handoff": copy.deepcopy(self.handoff_events[-1:]),
                "post_review_delivery": post_review,
            }
            return bounded_recovery(recovery, self._count_tokens, self.settings.history_reserve_tokens)
        recovery = {
            "previous_session_failure": previous.get("failure_reason"),
            "previous_sdk_tool_errors": previous.get("sdk_tool_errors", [])[-2:],
            "previous_handler_tool_errors": _recent_handler_errors(previous, handlers),
            "draft_counts": {"basic": len(draft.get("基本信息", {})),
                             "advice": len(draft.get("维修建议", [])),
                             "defects": len(draft.get("病害明细", []))},
            "working_notes": copy.deepcopy(state.get("working_notes", {})),
            "working_notes_are_evidence": False,
            "pending_workspace_retry": self._pending_reading_update(),
            "reading_length_recovery": self._reading_length_recovery(),
            "native_review_handoff": copy.deepcopy(self.handoff_events[-1:]),
        }
        return bounded_recovery(recovery, self._count_tokens, self.settings.history_reserve_tokens)

    def _continue_after_native_exit(self, exit_state):
        """Optional business feedback in the existing Agent; legacy default stops.

        Subclasses may return a short task instruction. This hook does not
        accept delivery, start another session, or replenish any budget.
        """
        return None

    def _generation_recovery_prompt(self, receipt):
        """Opt-in: subclasses supply accurate pending work for one stopped loop."""
        return None

    def _native_exit_is_settled(self):
        """Prove the latest reader response ended successfully before feedback."""
        if (getattr(self, "_single_agent_terminal_failure", None) is not None
                or not self._active_agent_call_id
                or any(row.get("status") != "settled" for row in self.model_dispatches)):
            return False
        calls = [row for row in self.client.calls if row.get("kind") == "agent_generation"]
        audits = [row for row in self.gateway_receipts if row.get("request_kind") == "agent_generation"]
        if not calls or not audits:
            return False
        call, audit = calls[-1], audits[-1]
        if (call.get("call_id") != self._active_agent_call_id
                or audit.get("call_id") != self._active_agent_call_id
                or call.get("request_id") != audit.get("request_id")
                or call.get("status") != "success" or audit.get("status") != "success"
                or call.get("http_status") != 200 or audit.get("http_status") != 200
                or call.get("done_received") is not True
                or call.get("finish_reason") != "stop" or audit.get("finish_reason") != "stop"):
            return False
        if not self._model_forward_lock.acquire(blocking=False):
            return False
        self._model_forward_lock.release()
        return True

    def _native_exit_continuation_config(self):
        # Existing workflows retain their historical exit behavior. A subclass
        # must explicitly implement business feedback to opt in.
        if type(self)._continue_after_native_exit is DocumentWorkflow._continue_after_native_exit:
            return {}
        def feedback(exit_state):
            if (self._finish_is_current() and exit_state.get("native_final_event") is True
                    and type(exit_state.get("response_text")) is str and exit_state["response_text"].strip()):
                return None
            return self._continue_after_native_exit(exit_state)
        return {"continue_after_native_exit": feedback,
                "native_exit_is_settled": self._native_exit_is_settled}

    def _parse_recovery_generation_is_settled(self):
        """Authorize error-history rendering only after a known HTTP result.

        SDK parse errors prove syntax failure, not that generation settled.
        Require the latest real client/gateway pair before a failed tool call
        can be shown as text in the next complete request.
        """
        origin = self._active_agent_call_id
        if (not origin or getattr(self, "_single_agent_terminal_failure", None) is not None
                or any(row.get("status") != "settled" for row in self.model_dispatches)):
            return False
        # Gateway receipts survive an explicit resume; compare them with the
        # equally durable client receipts, not only the new client's calls.
        # Old unfinished generations still require an observed stop below.
        calls = [row for row in [*getattr(self, "model_calls_before_resume", []), *self.client.calls]
                 if row.get("kind") == "agent_generation"]
        audits = [row for row in self.gateway_receipts if row.get("request_kind") == "agent_generation"]
        if (not calls or not audits or calls[-1].get("call_id") != origin
                or audits[-1].get("call_id") != origin
                or sum(row.get("call_id") == origin for row in calls) != 1
                or sum(row.get("call_id") == origin for row in audits) != 1):
            return False
        call, audit = calls[-1], audits[-1]
        usage = call.get("usage")
        if (call.get("status") != "success" or audit.get("status") != "success"
                or call.get("http_status") != 200 or audit.get("http_status") != 200
                or call.get("done_received") is not True or call.get("usage_known") is False
                or not call.get("request_id") or call["request_id"] != audit.get("request_id")
                or sum(row.get("request_id") == call["request_id"] for row in calls) != 1
                or sum(row.get("request_id") == call["request_id"] for row in audits) != 1
                or call.get("finish_reason") not in {"stop", "tool_calls", "function_call"}
                or call["finish_reason"] != audit.get("finish_reason")
                or not isinstance(usage, dict) or usage != audit.get("usage")
                or any(type(usage.get(key)) is not int or usage[key] < 0
                       for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
                or usage["total_tokens"] > self.client.max_model_len
                or usage["prompt_tokens"] != call.get("prompt_tokens_checked")
                or usage["prompt_tokens"] != audit.get("prompt_tokens_checked")):
            return False
        if any(row.get("generation_attempted", True) and not row.get("done_received")
               and not generation_stopped(row) for row in calls):
            return False
        known_ids = {row.get("call_id") for row in calls}
        if any(row.get("generation_attempted") and row.get("call_id") not in known_ids for row in audits):
            return False
        if not self._model_forward_lock.acquire(blocking=False):
            return False
        self._model_forward_lock.release()
        return True

    def _finish_is_current(self):
        delivery = self.workspace.delivery_status()
        if self.settings.single_document_agent and not self.last_finish_origin:
            return False
        binding = self._finish_binding(delivery)
        if self.settings.single_document_agent:
            snapshot = self.workspace.snapshot()
            if (not self.model_self_check or not self._artifact_read_coverage(snapshot)
                    or self.model_self_check.get("semantic_state_sha256") != self._semantic_identity(snapshot)):
                return False
            binding.update(file_sha256=self.artifact_delivery["file_sha256"],
                           content_identity_sha256=self.artifact_delivery.get("content_identity_sha256"),
                           semantic_state_sha256=self._semantic_identity(snapshot),
                           model_self_check_sha256=json_digest(self.model_self_check))
            # Finish keeps the audit version at submission. A later source read
            # or navigation note does not change the file or the semantic check.
            binding.pop("workspace_version")
            accepted = {key:value for key,value in (self.last_finish or {}).items() if key != "workspace_version"}
            return delivery["ready"] and accepted == binding
        return delivery["ready"] and self.last_finish == binding

    def _ensure_delivery_artifacts(self, status, reason=""):
        if not self.settings.single_document_agent:
            return
        snapshot = self.workspace.snapshot()
        if status != "completed":
            self.artifact_delivery = self.artifact_manager.export(snapshot, incomplete=True, reason=reason,
                structure_status={"phase":"incomplete", "complete":False,
                    "delivery":self.workspace.delivery_status(),"native_final":False})
        elif not self._finish_is_current():
            raise ValueError("Actual artifact completion requires current native finish and file self-check")
        record = copy.deepcopy(self.artifact_delivery)
        record.update(phase="completed" if status == "completed" else "incomplete",
            complete=(status == "completed"),incomplete=(status != "completed"),
            semantic_review={"status":"not_run_by_user_policy","performed":False,"passed":None},
            agent_file_self_check={"performed":bool(self.model_self_check),"host_semantic_verdict":None,
                "statement":copy.deepcopy(self.model_self_check)},
            actual_finish_origin=copy.deepcopy(self.last_finish_origin) if status == "completed" else None)
        atomic_json(Path(record["status_path"]),record)
        self.artifact_delivery = record

    def _validate_single_agent_native_delivery(self, outcome):
        self._require_stage_boundary(outcome)
        identifiers = outcome.get("reading_model_call_ids", [])
        if not identifiers or not self.last_finish_origin:
            raise ValueError("Continuous delivery needs actual finish and native-final generation receipts")
        if self.last_finish_origin["call_id"] not in identifiers:
            raise ValueError("Finish must originate from this actual continuous Agent session")
        if identifiers.index(self.last_finish_origin["call_id"]) >= len(identifiers) - 1:
            raise ValueError("Accepted finish needs a later actual native final generation")
        finish_audit = self._terminal_model_bundle(self.last_finish_origin["call_id"])["audit"]
        if any(self.last_finish_origin[key] != finish_audit[key] for key in
               ("request_id", "request_sha256", "response_sha256")):
            raise ValueError("Saved finish origin differs from its actual HTTP receipt")
        bundle = self._terminal_model_bundle(identifiers[-1])
        choices = bundle["response"].get("choices", [])
        if len(choices) != 1 or bundle["audit"].get("request_kind") != "agent_generation":
            raise ValueError("Native final must be the actual single Agent response")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if (choice.get("finish_reason") != "stop" or type(content) is not str or not content.strip()
                or choice.get("message", {}).get("tool_calls")
                or content.strip() != outcome.get("response_text")):
            raise ValueError("Native final event differs from its complete actual HTTP response")

    def _native_delivery_completed(self, outcome):
        return outcome.get("status") == "completed" and outcome.get("native_final_event") is True

    def _before_continuous_request(self, payload):
        """Subclass hook for receipt-driven continuation before request admission."""
        return payload

    def _before_native_model(self):
        """Optional progress feedback appended to actual SDK activity memory."""
        return None

    def _additional_reading_handlers(self):
        """Opt-in workflow tools; keep the ordinary document profile unchanged."""
        return {}

    def _register_initial_source(self, user_prompt, entry_mode, session_index, pending_initial):
        if self.source_delivery is not None and (
                session_index == 0 and entry_mode == "full" or pending_initial):
            self.source_delivery.register_initial_message(user_prompt,
                [{"evidence_id": unit["id"], "start": 0, "end": len(unit["text"])}
                 for unit in self.manifest["units"]])

    def run(self):
        entry_mode = self.previous_entry_mode
        finished = False
        try:
            if (self.settings.single_document_agent
                    and len(self.sessions) > self._invocation_session_start):
                raise RuntimeError("This single Agent invocation was already consumed; no automatic second SDK session")
            # Publishing may fail after an accepted model run. Re-export its
            # bound draft without spending new model/session/time budgets.
            if self.previous_status == "completed":
                if not self._finish_is_current():
                    raise ValueError("Completed checkpoint no longer matches its accepted draft")
                data = _parse_report(_compact(self.workspace.snapshot()["draft"]))
                self._validate_delivery_schema(data)
                self._ensure_delivery_artifacts("completed")
                return {"data": data, "receipt": self._receipt("completed", entry_mode)}
            self._check_budget()
            self._persist()
            if self.manifest.get("unverified_content"):
                if self.source_objects is None or self.source_objects.catalog["ledger_ready"] is not True:
                    raise ValueError("Manifest contains unsupported content without a complete source-object catalog; no generation permitted")
            if self.settings.require_semantic_review:
                self._drain_review_queue()
            elif self.pending_review is not None:
                raise ValueError("No-review mode cannot execute a persisted review queue")
            entry_mode, first_prompt = self._entry()
            self.entry_mode = entry_mode
            source_navigation_path = self.directory / "source-navigation-managed.json"
            workspace_navigation_path = self.directory / "workspace-navigation.json"
            self.workspace_tools.configure_managed_navigation(
                state=json.loads(workspace_navigation_path.read_bytes()) if workspace_navigation_path.exists() else None,
                on_progress=lambda state: atomic_json(workspace_navigation_path, state))
            source_handlers = self.tools.automatic_handlers(
                state=json.loads(source_navigation_path.read_bytes()) if source_navigation_path.exists() else None,
                on_progress=lambda state: atomic_json(source_navigation_path, state))
            handlers = {**source_handlers, "read_workspace": self.workspace_tools.summary,
                        "read_workspace_page": self.workspace_tools.read_page_managed,
                        "update_workspace": self._update_workspace, "review_draft": self._review_draft,
                        "finish": self._finish}
            if not self.settings.require_semantic_review:
                handlers.pop("review_draft")
            if self.settings.single_document_agent:
                handlers["inspect_delivery"] = self._inspect_delivery_managed
            handlers.update(self._additional_reading_handlers())
            if self.source_objects is not None:
                handlers.update({name: self._source_handler(function)
                    for name, function in self.source_objects.handlers().items()})
            handlers = {name: self._guard(function) for name, function in handlers.items()}
            session_stop = len(self.sessions) + 1 if self.settings.single_document_agent else self.settings.max_sessions
            for session_index in range(len(self.sessions), session_stop):
                self._check_budget()
                if not self.settings.single_document_agent and len(self.sessions) + len(self.review_sessions) >= self.settings.max_sessions:
                    break
                node = self._node("D2_tongagents_reading_session")
                if session_index == 0 and not self.reading_terminal_attempts:
                    user_prompt = first_prompt
                else:
                    state = self.workspace.snapshot()
                    recovery = self._reading_recovery(state, handlers)
                    # Full evidence and workspace remain available through
                    # tools. Recovery carries unresolved work, never replaces
                    # source evidence with a lossy report summary.
                    recovery_stage = ("当前草稿已审且交付就绪，请查看既有草稿后收尾；遵循 post_review_delivery 指引。\n"
                                      if recovery.get("post_review_delivery") else
                                      "继续同一报告。通过 read_workspace 查看最新事实、草稿和未解决问题，按需回读原文。\n")
                    user_prompt = recovery_stage + _compact({
                        "version": state["version"],
                        "counts": {key: len(state[key]) for key in ("facts", "relations", "questions")},
                        "draft_sha256": state["draft_sha256"], "draft_valid": state["draft_valid"],
                        "last_review_available": bool(self.review_feedback),
                        **recovery})
                    if self._count_chat_tokens([{"role": "system", "content": self._reader_system_prompt()}, {"role": "user", "content": user_prompt}]) + self.settings.planning_output_reserve + self.settings.tool_tokens > self.settings.max_model_len:
                        if self.settings.dynamic_output_budget:
                            raise ValueError("Complete recovery input occupies context; no navigation fallback permitted")
                        user_prompt = self._navigation_prompt() + "\n" + _compact(recovery)
                user_prompt = self._workspace_bootstrap_prompt(user_prompt, session_index)
                pending_initial = (self.source_delivery.pending_initial_messages()
                    if self.source_delivery is not None else [])
                if pending_initial and not (session_index == 0 and entry_mode == "full"):
                    # An explicit operator recovery must not turn an unreceived
                    # initial source into an assumed summary. Re-present the
                    # original text in a new exact message with current state.
                    user_prompt = ("\n\n".join(row["content"] for row in pending_initial)
                        + "\n\n当前恢复进度（原文仍需由本次请求实际接收）：\n" + user_prompt)
                with AuditedModelGateway(
                    self.client, deadline=self.deadline,
                    max_calls=max(1, self.settings.max_model_calls - sum(c.get("kind") in {"generation", "agent_generation"} for c in self._model_calls())),
                    max_output_tokens=self.settings.output_tokens,
                    before_request=self._check_budget,
                    forward_lock=self._model_forward_lock,
                    recovery_prompt=(self._generation_recovery_prompt
                        if type(self)._generation_recovery_prompt is not DocumentWorkflow._generation_recovery_prompt else None),
                ) as gateway:
                    self._active_agent_call_id = None
                    first_audit = len(self.gateway_receipts)
                    gateway.receipts = self.gateway_receipts
                    gateway.on_dispatch = self._on_dispatch
                    gateway.on_prepared_request = self._on_model_request
                    gateway.on_input_observed = self._source_input_observed
                    gateway.on_response = self._on_model_response
                    gateway.on_receipt = self._on_model_receipt
                    self._register_initial_source(user_prompt, entry_mode, session_index, pending_initial)
                    if self.source_delivery is None and session_index == 0 and entry_mode == "full":
                        def observed_full(_payload, _receipt):
                            if not any(isinstance(m.get("content"), str) and first_prompt in m["content"]
                                       for m in _payload["messages"]):
                                return
                            for unit in self.manifest["units"]:
                                self.workspace.record_read(unit["id"], 0, len(unit["text"]))
                            gateway.on_success = None
                        gateway.on_success = observed_full
                    config = {**self.react_config, "url": gateway.url, "system_prompt": self._reader_system_prompt(),
                              "enable_thinking": self.settings.enable_thinking,
                              "temperature": self.settings.temperature,
                              "current_agent_call_id": lambda: self._active_agent_call_id,
                              "max_tokens": self.settings.output_tokens, "max_iterations": (1048576 if self.settings.single_document_agent else self.settings.session_iterations),
                              "max_tool_tokens": self.settings.tool_tokens, "count_tokens": self._count_tokens,
                              "max_calls": (1048576 if self.settings.single_document_agent else self.settings.session_calls),
                              "timeout": min(self.react_config.get("timeout", 600), max(0.01, self.deadline - time.monotonic()))}
                    if self.settings.single_document_agent:
                        config["on_tool_result"] = self._observe_native_tool
                        config["same_agent_parse_recovery"] = True
                        config["before_same_agent_continue"] = self._check_budget
                        config["before_native_model"] = self._before_native_model
                        config["select_generation_tools"] = self._before_continuous_request
                        config["parse_recovery_generation_is_settled"] = self._parse_recovery_generation_is_settled
                        config.update(self._native_exit_continuation_config())
                        config["max_history_len"] = 1048576
                        from .document_history_continuity import HistoryPrefixGuard
                        history_guard = HistoryPrefixGuard()
                        if self.settings.native_context_compression:
                            from .document_context_compression import NativeContextCompression
                            def refresh_exact_progress():
                                self.context_state.observe('before_native_compression', {}, {'ok':True}, self.workspace.snapshot())
                            self.native_compression = NativeContextCompression(self.directory / 'native_context_compression',
                                summary_client=self.compression_client, exact_progress=self.context_state,
                                before_model=self._check_budget, refresh_progress=refresh_exact_progress,
                                history_guard=history_guard, trigger_tokens=self.settings.compression_trigger_tokens,
                                deadline=self.deadline, identity=self.identity)
                            config['native_compression'] = self.native_compression
                        def prepare_continuous(payload):
                            self._check_budget()
                            try:
                                if history_guard.snapshot()["observations"] == 0:
                                    messages = payload.get("messages", []) if type(payload) is dict else []
                                    if (type(messages) is not list or
                                            not any(type(m) is dict and m.get("role") == "system"
                                                and type(m.get("content")) is str
                                                and self._reader_system_prompt() in m["content"] for m in messages) or
                                            not any(type(m) is dict and m.get("role") == "user"
                                                and m.get("content") == user_prompt for m in messages)):
                                        raise InitialSourceDeliveryError("First actual SDK request must preserve complete supplied SYSTEM and USER")
                                continuity = history_guard.observe_complete_request(payload)
                                if self.native_compression is not None:
                                    self.native_compression.on_admitted_transition(continuity)
                            except Exception as exc:
                                self.client._dynamic_budget_failure = type(exc).__name__
                                raise
                            finally:
                                atomic_json(self.directory / "history_continuity" / f"session-{session_index + 1:03d}.json",
                                            {"workflow_identity_sha256": json_digest(self.identity),
                                             "session_index": session_index, "receipts": history_guard.receipts,
                                             "state": history_guard.snapshot(),
                                             "required_system_prompt_sha256": json_digest(self._reader_system_prompt()),
                                             "required_user_prompt_sha256": json_digest(user_prompt),
                                             "first_prompt_binding_accepted": bool(history_guard.snapshot()["observations"]),
                                             "admission_failure": getattr(self.client, "_dynamic_budget_failure", None)})
                            prepared = gateway.prepare_request(payload)
                            if self.native_compression is not None:
                                # prepare_request is exact complete-wire admission; no extra tokenization or truncation.
                                self.native_compression.observe_complete_request(prepared, gateway.last_prepared_allocation['actual_input_tokens'])
                            return prepared
                    runner_error = None
                    try:
                        if self.settings.dynamic_output_budget:
                            config.update(dynamic_output_budget=True, prepare_generation_request=(prepare_continuous if self.settings.single_document_agent else gateway.prepare_request))
                        outcome = self.react_runner(user_prompt, handlers, config)
                    except Exception as exc:
                        # The native session was actually started. Consume its
                        # existing session allowance even without an SDK outcome;
                        # do not claim native finality or successful tool actions.
                        runner_error = exc
                        outcome = {"status": "failed", "native_final_event": False,
                                   "native_outcome_received": False,
                                   "failure_reason": "reading_runner_exception",
                                   "error_type": type(exc).__name__, "tool_calls": []}
                    finally:
                        self._active_agent_call_id = None
                        try:
                            terminal_attempts = self._capture_reading_terminal_attempts(
                                gateway.receipts[first_audit:], session_index)
                        except Exception as exc:
                            # Provenance failure must still consume this started
                            # session; it never creates a length classification.
                            terminal_attempts = []
                            if runner_error is None:
                                runner_error = exc
                            outcome = {**outcome, "status": "failed",
                                       "failure_reason": "reading_terminal_provenance_failed",
                                       "terminal_provenance_error": type(exc).__name__}
                    if (not self.settings.dynamic_output_budget and session_index == 0 and entry_mode == "full" and any(
                            a.get("prompt_tokens_checked", 0) + self.settings.planning_output_reserve > self.settings.max_model_len
                            for a in gateway.receipts[first_audit:])):
                        entry_mode = self.entry_mode = "navigation"
                outcome["reading_model_call_ids"] = [a["call_id"] for a in gateway.receipts[first_audit:]
                    if a.get("request_kind") == "agent_generation" and a.get("status") == "success"]
                outcome["reading_terminal_attempts"] = terminal_attempts
                self.sessions.append(outcome)
                node.update(status=outcome["status"], native_final_event=outcome.get("native_final_event"),
                            sdk_tool_calls=len(outcome.get("tool_calls", [])))
                self._persist()
                if runner_error is not None:
                    raise runner_error
                if self.settings.require_semantic_review:
                    self._drain_review_queue()
                if self._finish_is_current():
                    if not self._native_delivery_completed(outcome):
                        continue
                    if not any(t.get("name") == self._native_finish_tool_name() and t.get("status") == "success" for t in outcome.get("tool_calls", [])):
                        continue
                    if not any(c.get("kind") == "agent_generation" and c.get("status") == "success" for c in self._model_calls()):
                        continue
                    if self.settings.single_document_agent:
                        self._validate_single_agent_native_delivery(outcome)
                    self._check_budget(include_model=False)
                    finished = True
                    break
            if not finished:
                if self.settings.single_document_agent:
                    raise RuntimeError("Single document Agent ended without accepted native delivery; progress saved, no automatic new session")
                raise RuntimeError("Reading session budget exhausted before accepted delivery")
            data = _parse_report(_compact(self.workspace.snapshot()["draft"]))
            self._ensure_delivery_artifacts("completed")
            receipt = self._receipt("completed", entry_mode)
            self._persist("completed")
            return {"data": data, "receipt": receipt}
        except Exception as exc:
            try:
                self._ensure_delivery_artifacts("failed",type(exc).__name__+": "+str(exc))
            except Exception as artifact_error:
                atomic_json(self.directory / "delivery-export-error.json",
                    {"status":"incomplete","complete":False,"error_type":type(artifact_error).__name__,
                     "original_failure_type":type(exc).__name__,"workspace_checkpoint":str(self.workspace.checkpoint_path),
                     "semantic_verdict":None})
            receipt = self._receipt("failed", entry_mode)
            receipt["error"] = type(exc).__name__
            self._persist("failed")
            raise DocumentWorkflowFailure(str(exc), receipt) from exc

    def _receipt(self, status, entry_mode):
        snapshot = self.workspace.snapshot()
        calls = self._model_calls()
        result = {"schema_version": 1, "workflow": VERSION, "status": status,
                "identity": self.identity, "entry_mode": entry_mode, "dag_nodes": self.nodes,
                "sdk_sessions": self.sessions, "native_review_sessions": self.review_sessions,
                "pending_review": copy.deepcopy(self.pending_review), "handoff_events": self.handoff_events,
                "review_engine": {"FORWARD": "native_tongagents", "REVERSE": "native_tongagents",
                                  "source_object_dispositions": "existing_gateway_review"},
                "tool_calls": self._tool_calls(),
                "reading_terminal_attempts": copy.deepcopy(self.reading_terminal_attempts),
                "model_calls": calls, "agent_generation_requests": sum(c.get("kind") == "agent_generation" for c in calls),
                "review_generation_requests": sum(c.get("kind") == "generation" and c.get("purpose") != "image_observation" for c in calls),
                "image_generation_requests": sum(c.get("kind") == "generation" and c.get("purpose") == "image_observation" for c in calls),
                "gateway_receipts": self.gateway_receipts,
                "model_dispatches": copy.deepcopy(self.model_dispatches),
                "uncertain_generation_attempts": sum(d.get("status") == "outcome_unknown" for d in self.model_dispatches),
                "uncertain_budget_seconds": self.uncertain_budget_seconds,
                "workspace_version": snapshot["version"], "draft_sha256": snapshot["draft_sha256"],
                "final_data": snapshot["draft"], "delivery": self.workspace.delivery_status(),
                "seconds": self.total_spent_seconds + time.monotonic() - self.started,
                "knowledge_enabled": False, "evolution_enabled": False}
        if self.settings.single_document_agent:
            result["review_engine"] = {"status": "not_run_by_user_policy", "performed": False, "passed": None}
            result["single_document_agent"] = {"automatic_new_session": False, "context_compression": self.settings.native_context_compression,
                "native_session_invocations_this_run": len(self.sessions) - getattr(self, "_invocation_session_start", 0),
                "explicit_operator_resume": self.explicit_operator_resume,
                "restored_context_from_old_agent": False,
                "local_positive_limit": 1048576, "actual_global_model_call_limit": self.settings.max_model_calls}
            result['native_compression'] = {'enabled':self.settings.native_context_compression,
                'trigger_tokens':self.settings.compression_trigger_tokens,
                'transitions':copy.deepcopy(self.native_compression.transitions) if self.native_compression else [],
                'summary_model_calls':copy.deepcopy(getattr(self.compression_client,'calls',[]))}
            result["accepted_finish"] = copy.deepcopy(self.last_finish)
            result["actual_finish_origin"] = copy.deepcopy(self.last_finish_origin)
            result["model_self_check"] = copy.deepcopy(self.model_self_check)
            result["artifact_delivery"] = copy.deepcopy(self.artifact_delivery)
            result["delivery_inspections"] = copy.deepcopy(self.delivery_inspections)
        if self.source_objects is not None:
            result["source_objects"] = self._source_status(snapshot)
            result["accepted_finish"] = copy.deepcopy(self.last_finish)
        return result
