from wine_quality.components.reporting import ProjectReporter
from wine_quality.configuration.configuration import ConfigurationManager


class ReportingPipeline:
    def main(self):
        config = ConfigurationManager()
        reporting_config = config.get_reporting_config()
        reporter = ProjectReporter(config=reporting_config)
        reporter.create_visuals()
