"""Console widget for displaying stdout and stderr output."""

import logging
import sys
from io import StringIO

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class OutputStream(QObject):
    """Custom output stream that emits signals when text is written."""

    text_written = Signal(str)

    def __init__(self, parent):
        """Initialize the output stream."""
        super().__init__(parent)
        self._buffer = StringIO()

    def write(self, text: str):
        """Write text to the output stream."""
        if text:
            self._buffer.write(text)
            self.text_written.emit(text)

    def flush(self):
        """Flush the output stream."""
        self._buffer.flush()

    def reset(self):
        """Reset the output stream."""
        self._buffer = StringIO()


class LoggingSignalHandler(logging.Handler):
    """Logging signal handler for the output stream."""

    def __init__(self, stream: OutputStream):
        """Initialize the logging signal handler."""
        super().__init__()
        self._stream = stream

    def emit(self, record: logging.LogRecord):
        """Emit the logging record."""
        msg = self.format(record)
        self._stream.write(msg + "\n")


class ConsoleWidget(QWidget):
    """Console widget for displaying command line output."""

    def __init__(self, parent):
        """Initialize the console widget."""
        super().__init__(parent)
        self.setVisible(False)
        self._stdout_stream = None
        self._stderr_stream = None
        self._original_stdout = None
        self._original_stderr = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(5, 0, 5, 0)

        title_label = QLabel("Log")
        title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFontFamily("Consolas, Courier New, monospace")
        self.text_edit.setFontPointSize(10)
        layout.addWidget(self.text_edit, 1)

        self._setup_output_redirection()
        self._setup_logging_redirection()

    def _setup_output_redirection(self):
        self._stdout_stream = OutputStream(self)
        self._stderr_stream = OutputStream(self)

        self._stdout_stream.text_written.connect(self._append_stdout)
        self._stderr_stream.text_written.connect(self._append_stderr)

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self._stdout_stream
        sys.stderr = self._stderr_stream

    def _setup_logging_redirection(self):
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for h in root.handlers:
            if isinstance(h, LoggingSignalHandler):
                root.removeHandler(h)
        self._logging_handler = LoggingSignalHandler(self._stdout_stream)
        root.addHandler(self._logging_handler)

    def _append_stdout(self, text: str):
        """Append stdout text."""
        self._append_text(text, None)

    def _append_stderr(self, text: str):
        """Append stderr text in red."""
        color = QColor("#cc0000")
        self._append_text(text, color)

    def _append_text(self, text: str, color: QColor):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if color:
            text_format = cursor.charFormat()
            text_format.setForeground(color)
            cursor.setCharFormat(text_format)

        cursor.insertText(text)

        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def showEvent(self, event):
        """Show the console widget."""
        super().showEvent(event)
        if sys.stdout != self._stdout_stream:
            self._setup_output_redirection()

    def closeEvent(self, event):
        """Close the console widget."""
        self._restore_output_streams()
        super().closeEvent(event)

    def _restore_output_streams(self):
        """Restore the output streams."""
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr
