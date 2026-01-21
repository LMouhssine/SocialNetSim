# Contributing to SocialNetSim

Thank you for your interest in contributing to SocialNetSim! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)
- Any relevant logs or screenshots

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature has already been requested
- Clearly describe the feature and its use case
- Explain why this would be useful to the project

### Submitting Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Follow the existing code style
   - Add comments where necessary

3. **Add tests**:
   - All new features should have tests
   - Ensure existing tests still pass
   - Aim for good test coverage

4. **Run the test suite**:
   ```bash
   pytest tests/
   ```

5. **Format your code**:
   ```bash
   black .
   flake8 .
   ```

6. **Commit your changes**:
   - Write clear, descriptive commit messages
   - Reference issues if applicable (#123)
   ```bash
   git commit -m "Add feature: description of what you did"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**:
   - Provide a clear description of your changes
   - Link to any related issues
   - Explain the motivation and context

## Code Style

- Follow PEP 8 guidelines
- Use [Black](https://github.com/psf/black) for code formatting
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized

### Example

```python
def calculate_engagement_probability(
    user: User,
    post: Post,
    state: SimulationState,
) -> float:
    """Calculate the probability a user will engage with a post.

    Args:
        user: The user viewing the post
        post: The post being viewed
        state: Current simulation state

    Returns:
        Engagement probability between 0 and 1
    """
    # Implementation here
    pass
```

## Testing

- Write unit tests for new functions
- Write integration tests for new features
- Ensure all tests pass before submitting
- Use pytest fixtures for common test setup

## Documentation

- Update README.md if you change functionality
- Add docstrings to new functions and classes
- Update type hints when modifying function signatures
- Comment complex algorithms or non-obvious code

## Project Structure

Familiarize yourself with the project structure:

```
SocialNetSim/
├── config/         # Configuration schemas and scenarios
├── models/         # Data models (User, Post, etc.)
├── generator/      # Synthetic data generation
├── engine/         # Simulation engine components
├── ai/             # Machine learning models
├── scenarios/      # Experiment framework
├── ui/             # Streamlit dashboard
├── scripts/        # Command-line tools
└── tests/          # Test suite
```

## Questions?

If you have questions about contributing, feel free to:
- Open a [Discussion](https://github.com/LMouhssine/SocialNetSim/discussions)
- Comment on an existing issue
- Reach out via [Issues](https://github.com/LMouhssine/SocialNetSim/issues)

## Code of Conduct

We expect all contributors to:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the project
- Show empathy towards other community members

## License

By contributing to SocialNetSim, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making SocialNetSim better! 🎉
