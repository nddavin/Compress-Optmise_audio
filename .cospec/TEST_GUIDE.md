## 测试指导文档

### 测试命令使用说明

#### 基础测试
```bash
# 运行所有单元测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_compress_audio.py -v
python -m pytest tests/test_new_system.py -v

# 运行测试并生成覆盖率报告
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# 运行测试并显示详细输出
python -m pytest tests/ -v --tb=short

# 运行测试并并行执行
python -m pytest tests/ -n auto
```

#### Docker环境测试
```bash
# 使用Docker Compose运行测试
./scripts/start.sh test

# 进入开发容器并运行测试
./scripts/start.sh dev
# 在容器内运行:
python -m pytest tests/ -v

# 运行特定测试模块
python -m pytest tests/test_compress_audio.py -v
```

#### 按功能模块测试
```bash
# 测试音频压缩功能
python -m pytest tests/test_compress_audio.py -k "test_get_format_defaults" -v

# 测试事件系统和依赖注入
python -m pytest tests/test_new_system.py -k "TestEventSystem" -v

# 测试依赖注入容器
python -m pytest tests/test_new_system.py -k "TestDIContainer" -v

# 测试接口标准化
python -m pytest tests/test_new_system.py -k "TestInterfaces" -v
```

#### 测试配置和过滤
```bash
# 只运行失败的用例
python -m pytest tests/ --lf

# 不运行标记为skip的用例
python -m pytest tests/ -m "not skip"

# 运行特定标记的用例
python -m pytest tests/ -m "integration"

# 详细输出调试信息
python -m pytest tests/ -v --tb=long

# 显示每个测试的执行时间
python -m pytest tests/ -v --durations=10
```

### 测试案例管理规范

#### 测试文件组织结构
```
tests/
├── test_compress_audio.py      # 音频压缩核心功能测试
├── test_new_system.py          # 事件系统、依赖注入、接口测试
├── conftest.py                 # 测试配置文件
├── pytest.ini                 # pytest配置
├── __init__.py                # 测试包初始化
└── data/                      # 测试数据目录
    ├── sample_audio/
    ├── test_configs/
    └── expected_outputs/
```

#### 测试用例命名规范
- **文件命名**: `test_<模块名>.py`
- **测试类命名**: `Test<模块名>CamelCase`
- **测试方法命名**: `test_<功能描述>_snake_case`

#### 测试用例分类
1. **单元测试** (Unit Tests)
   - 测试单个函数、类或方法
   - 快速执行，隔离依赖
   - 示例: `test_get_format_defaults_mp3_speech()`

2. **集成测试** (Integration Tests)
   - 测试模块间交互
   - 需要真实环境依赖
   - 示例: `test_event_driven_di_container()`

3. **接口测试** (Interface Tests)
   - 验证标准化接口实现
   - 测试数据传输对象
   - 示例: `test_audio_processor_interface()`

#### 测试数据管理
- **测试音频文件**: 存放在 `tests/data/sample_audio/` 目录
- **配置文件**: 存放在 `tests/data/test_configs/` 目录
- **预期输出**: 存放在 `tests/data/expected_outputs/` 目录
- **临时文件**: 使用 `tempfile` 模块创建，确保测试后清理

#### 测试工具配置
```python
# conftest.py 示例配置
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_audio_dir():
    """创建临时音频目录的fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_audio_file():
    """返回示例音频文件路径"""
    return Path(__file__).parent / "data" / "sample_audio" / "test.wav"

@pytest.fixture
def mock_event_bus():
    """返回模拟事件总线"""
    from unittest.mock import Mock
    return Mock()
```

#### 测试断言规范
- 使用 `unittest.TestCase` 或 `pytest` 的断言风格
- 提供清晰的错误信息
- 验证边界条件和异常情况
- 示例:
```python
def test_compression_ratio(self):
    result = compress_audio(input_path, output_path, bitrate=128)
    self.assertGreaterEqual(result.compression_ratio, 0.5)  # 至少50%压缩率
    self.assertLessEqual(result.compression_ratio, 1.0)    # 不能超过100%
```

#### 测试覆盖率要求
- **核心模块**: 压缩功能 ≥ 90%
- **事件系统**: ≥ 95%
- **依赖注入**: ≥ 90%
- **接口实现**: ≥ 85%
- **整体项目**: ≥ 80%

#### 测试执行顺序
1. **快速单元测试** (1-2秒)
2. **集成测试** (10-30秒)
3. **接口测试** (5-10秒)
4. **性能测试** (可选，1-5分钟)

#### 测试环境配置
- **Python版本**: 3.8+
- **FFmpeg**: 必须安装并可用
- **依赖库**: pytest, pytest-cov, pytest-mock
- **测试隔离**: 每个测试用例独立运行，避免状态污染

#### 测试报告生成
```bash
# 生成HTML覆盖率报告
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# 生成JUnit XML报告
python -m pytest tests/ --junitxml=test_results.xml

# 生成覆盖率报告
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

#### 常见测试问题解决
1. **FFmpeg未找到**: 确保FFmpeg在PATH中或使用绝对路径
2. **测试数据缺失**: 在 `tests/data/` 目录中添加必要的测试文件
3. **依赖冲突**: 使用虚拟环境隔离测试依赖
4. **测试超时**: 调整pytest的timeout设置或优化测试逻辑

#### 扩展测试
1. **性能测试**: 添加大文件处理性能基准测试
2. **端到端测试**: GUI界面自动化测试
3. **负载测试**: 并发处理能力测试
4. **故障恢复测试**: 异常情况处理能力测试

#### 测试最佳实践
- 使用 `@pytest.fixture` 管理测试资源
- 遵循AAA模式 (Arrange-Act-Assert)
- 编写可读性强的测试用例
- 定期更新测试数据
- 保持测试代码简洁，避免过度设计