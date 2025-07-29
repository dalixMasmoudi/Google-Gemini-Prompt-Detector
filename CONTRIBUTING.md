# Contributing to GuardiaVision Gemini

Thank you for your interest in contributing to GuardiaVision Gemini! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Google API key for testing
- Basic understanding of computer vision concepts

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/guardiavision-gemini.git
   cd guardiavision-gemini
   ```

2. **Set up development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to ensure everything works**
   ```bash
   python test_modular.py
   pytest tests/
   ```

## 🛠 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Docstrings**: Google style docstrings for all public functions/classes
- **Type hints**: Required for all public APIs

### Code Formatting

We use automated code formatting:

```bash
# Format code
black src/ main.py

# Sort imports
isort src/ main.py

# Check style
flake8 src/ main.py

# Type checking
mypy src/
```

### Documentation

- All public functions and classes must have comprehensive docstrings
- Include parameter descriptions, return values, and examples
- Update README.md for new features
- Add inline comments for complex logic

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Include both unit tests and integration tests
- Test edge cases and error conditions

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detector.py
```

## 📝 Contribution Types

### Bug Reports

When reporting bugs, please include:

- **Environment**: Python version, OS, dependencies
- **Steps to reproduce**: Minimal example that demonstrates the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full stack traces if applicable
- **Images**: Sample images that cause issues (if relevant)

### Feature Requests

For new features, please provide:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Breaking changes**: Will this affect existing code?

### Code Contributions

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   python test_modular.py
   pytest
   black --check src/
   flake8 src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new detection feature"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   # Create PR on GitHub
   ```

## 🏗 Architecture Guidelines

### Module Structure

When adding new modules:

- Place in appropriate `src/` subdirectory
- Follow existing naming conventions
- Include comprehensive docstrings
- Add to `src/__init__.py` if public API

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Fail gracefully when possible

### Performance

- Profile code for performance bottlenecks
- Use appropriate data structures
- Consider memory usage for large images
- Add performance tests for critical paths

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Test Categories

1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Measure timing and resource usage

### Writing Tests

```python
import pytest
from src.detector import GeminiDetector

class TestGeminiDetector:
    def test_initialization(self):
        """Test detector initialization."""
        detector = GeminiDetector(api_key="test_key")
        assert detector.model_name == "gemini-2.5-flash-preview-05-20"
    
    def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        with pytest.raises(ValueError):
            GeminiDetector(api_key="")
```

## 📋 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests and checks
2. **Code review**: Maintainers review code quality and design
3. **Testing**: Reviewers test functionality
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

## 🐛 Debugging Guidelines

### Common Issues

1. **API Key Problems**
   - Verify key is set correctly
   - Check key permissions
   - Test with simple API call

2. **Image Processing Issues**
   - Check image format support
   - Verify image file exists
   - Test with different image sizes

3. **Detection Accuracy**
   - Try different prompts
   - Adjust confidence thresholds
   - Check image quality

### Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use performance monitoring
from src.utils import PerformanceMonitor
monitor = PerformanceMonitor()

# Add debug prints
print(f"Debug: {variable_name}")
```

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Release notes prepared
- [ ] Tagged in Git

## 💬 Communication

### Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and discussion

### Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## 🎯 Roadmap

### Current Priorities

1. **Performance Optimization**: Faster processing times
2. **Additional Models**: Support for other vision models
3. **Batch Processing**: Process multiple images efficiently
4. **Web Interface**: Browser-based interface
5. **Cloud Integration**: Deploy to cloud platforms

### How to Help

- Check the [Issues](https://github.com/guardiavision/guardiavision-gemini/issues) page
- Look for "good first issue" labels
- Contribute to documentation
- Report bugs and suggest improvements
- Share the project with others

## 📚 Resources

- [Python Style Guide](https://pep8.org/)
- [Google Docstring Style](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Git Best Practices](https://git-scm.com/book)

Thank you for contributing to GuardiaVision Gemini! 🙏
