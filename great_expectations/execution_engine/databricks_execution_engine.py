import datetime
import logging
from typing import Any, Callable, Tuple

from great_expectations.core.batch import BatchMarkers
from great_expectations.core.batch_spec import (
    AzureBatchSpec,
    BatchSpec,
    PathBatchSpec,
    RuntimeDataBatchSpec,
)
from great_expectations.core.util import AzureUrl
from great_expectations.exceptions import exceptions as ge_exceptions
from great_expectations.execution_engine import SparkDFExecutionEngine

from ..exceptions import BatchSpecError, ExecutionEngineError
from .sparkdf_batch_data import SparkDFBatchData

logger = logging.getLogger(__name__)

try:
    import pyspark
    import pyspark.sql.functions as F

    # noinspection SpellCheckingInspection
    import pyspark.sql.types as sparktypes
    from pyspark import SparkContext
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.readwriter import DataFrameReader
except ImportError:
    pyspark = None
    SparkContext = None
    SparkSession = None
    DataFrame = None
    DataFrameReader = None
    F = None
    # noinspection SpellCheckingInspection
    sparktypes = None

    logger.debug(
        "Unable to load pyspark; install optional spark dependency for support."
    )


class DatabricksDBFSSparkDFExecutionEngine(SparkDFExecutionEngine):
    """
    Use this class for interacting with data on DBFS when using Databricks Spark clusters.
    """

    def get_batch_data_and_markers(
        self, batch_spec: BatchSpec
    ) -> Tuple[Any, BatchMarkers]:  # batch_data
        # We need to build a batch_markers to be used in the dataframe
        batch_markers: BatchMarkers = BatchMarkers(
            {
                "ge_load_time": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y%m%dT%H%M%S.%fZ"
                )
            }
        )

        """
        As documented in Azure DataConnector implementations, Pandas and Spark execution engines utilize separate path
        formats for accessing Azure Blob Storage service.  However, Pandas and Spark execution engines utilize identical
        path formats for accessing all other supported cloud storage services (AWS S3 and Google Cloud Storage).
        Moreover, these formats (encapsulated in S3BatchSpec and GCSBatchSpec) extend PathBatchSpec (common to them).
        Therefore, at the present time, all cases with the exception of Azure Blob Storage , are handled generically.
        """

        batch_data: Any
        if isinstance(batch_spec, RuntimeDataBatchSpec):
            # batch_data != None is already checked when RuntimeDataBatchSpec is instantiated
            batch_data = batch_spec.batch_data
            if isinstance(batch_data, str):
                raise ge_exceptions.ExecutionEngineError(
                    f"""SparkDFExecutionEngine has been passed a string type batch_data, "{batch_data}", which is illegal.
    Please check your config."""
                )
            batch_spec.batch_data = "SparkDataFrame"

        elif isinstance(batch_spec, AzureBatchSpec):
            reader_method: str = batch_spec.reader_method
            reader_options: dict = batch_spec.reader_options or {}
            path: str = batch_spec.path
            azure_url = AzureUrl(path)
            try:
                credential = self._azure_options.get("credential")
                storage_account_url = azure_url.account_url
                if credential:
                    self.spark.conf.set(
                        "fs.wasb.impl",
                        "org.apache.hadoop.fs.azure.NativeAzureFileSystem",
                    )
                    self.spark.conf.set(
                        "fs.azure.account.key." + storage_account_url, credential
                    )
                reader: DataFrameReader = self.spark.read.options(**reader_options)
                reader_fn: Callable = self._get_reader_fn(
                    reader=reader,
                    reader_method=reader_method,
                    path=path,
                )
                # For usage with DBFS in Databricks.
                # File based Dataconnectors are used with the `/dbfs` prepended to the path,
                # but the databricks spark df reader methods already translate to dbfs:/ paths
                if path.startswith("/dbfs"):
                    path = path.replace("/dbfs", "", 1)
                batch_data = reader_fn(path)
            except AttributeError:
                raise ExecutionEngineError(
                    """
                    Unable to load pyspark. Pyspark is required for SparkDFExecutionEngine.
                    """
                )

        elif isinstance(batch_spec, PathBatchSpec):
            reader_method: str = batch_spec.reader_method
            reader_options: dict = batch_spec.reader_options or {}
            path: str = batch_spec.path
            try:
                reader: DataFrameReader = self.spark.read.options(**reader_options)
                reader_fn: Callable = self._get_reader_fn(
                    reader=reader,
                    reader_method=reader_method,
                    path=path,
                )
                # For usage with DBFS in Databricks.
                # File based Dataconnectors are used with the `/dbfs` prepended to the path,
                # but the databricks spark df reader methods already translate to dbfs:/ paths
                if path.startswith("/dbfs"):
                    path = path.replace("/dbfs", "", 1)
                batch_data = reader_fn(path)
            except AttributeError:
                raise ExecutionEngineError(
                    """
                    Unable to load pyspark. Pyspark is required for SparkDFExecutionEngine.
                    """
                )
            # pyspark will raise an AnalysisException error if path is incorrect
            except pyspark.sql.utils.AnalysisException:
                raise ExecutionEngineError(
                    f"""Unable to read in batch from the following path: {path}. Please check your configuration."""
                )

        else:
            raise BatchSpecError(
                """
                Invalid batch_spec: batch_data is required for a SparkDFExecutionEngine to operate.
                """
            )

        batch_data = self._apply_splitting_and_sampling_methods(batch_spec, batch_data)
        typed_batch_data = SparkDFBatchData(execution_engine=self, dataframe=batch_data)

        return typed_batch_data, batch_markers
