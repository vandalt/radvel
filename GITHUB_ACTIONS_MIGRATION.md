# Migration from Travis CI to GitHub Actions

This guide explains how to migrate your radvel CI/CD pipeline from Travis CI to GitHub Actions.

## What's Been Created

A GitHub Actions workflow file has been created:

**`.github/workflows/ci.yml`** - Simple, reliable workflow using pip (faster and more reliable than conda)

## Required Setup

### 1. Repository Secrets

You'll need to add these secrets in your GitHub repository settings:

- **`COVERALLS_REPO_TOKEN`**: Your Coveralls repository token
- **`PYPI_TOKEN`**: Your PyPI API token for publishing packages

To add secrets:
1. Go to your repository on GitHub
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each secret with the appropriate name and value

### 2. Workflow Details

The workflow uses pip for dependency management, which provides:
- Faster builds
- More reliable dependency resolution
- Better compatibility with GitHub Actions
- Simpler setup and maintenance

## What the Workflows Do

### Test Job
- Runs on Python 3.8, 3.9, 3.10, and 3.11
- Installs system dependencies (gcc, git, pkg-config, libhdf5)
- Installs Python dependencies
- Builds the package with C extensions
- Runs tests with coverage
- Uploads coverage to Coveralls (only on Python 3.8)

### Deploy Job
- Only runs when you create a tag (e.g., `v1.0.0`)
- Builds the package distribution
- Publishes to PyPI

## Migration Steps

1. **Add repository secrets** for Coveralls and PyPI
2. **Test the workflow** by pushing to a branch or creating a PR
3. **Remove Travis CI** once you're satisfied with GitHub Actions

## Benefits of GitHub Actions

- **Faster builds**: No Docker layer caching delays
- **Better matrix support**: Easier to test multiple Python versions
- **Native GitHub integration**: Better PR integration, status checks
- **Free tier**: More generous than Travis CI's free tier
- **Modern CI/CD**: Better security, caching, and workflow features

## Troubleshooting

If you encounter issues:

1. Check the Actions tab in your GitHub repository
2. Look at the workflow run logs for specific error messages
3. Ensure all required secrets are properly configured
4. Verify that your `requirements.txt` and `setup.py` are compatible with the Python versions being tested

## Next Steps

After migration:
1. Update your README.md to remove Travis CI badges
2. Consider adding GitHub Actions status badges
3. Test the deployment process with a test tag
4. Remove the `.travis.yml` file once everything is working
