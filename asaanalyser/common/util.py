# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import itertools
import csv, codecs, cStringIO

def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")

def total_seconds(td):
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / float(10**6)

def td_to_str(td, use_ms=True):
    mm, ss = divmod(td.seconds, 60)
    hh, mm = divmod(mm, 60)
    if td.days:
        hh += td.days*24
    s = "%d:%02d:%02d" % (hh, mm, ss)
    if td.microseconds and use_ms:
        s += ".%06d" % td.microseconds
    return s

def seconds_to_td_str(seconds, use_ms=True):
    return td_to_str(timedelta(seconds=seconds), use_ms)

class CSVUnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """
    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([unicode(s).encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            row = [v if not v is None else "" for v in row]
            row = [v if not type(v) is timedelta else td_to_str(v) for v in row]
            self.writerow(row)

def xls_to_lists(workbook_path):
    """
    Receives the path to an excel file, which it transforms to a dict
    of lists of lists (one dict entry per whorksheet)
    """
    csv_data = {}

    import xlrd
    wb = xlrd.open_workbook(workbook_path)
    for sh_name in wb.sheet_names():
        sh = wb.sheet_by_name(sh_name)

        headers = sh.row_values(0)
        data = []
        for rownum in xrange(1, sh.nrows):
            data.append(sh.row_values(rownum))
        csv_data[sh_name] = (headers, data)
    return csv_data

def groupby(tuples_iterable, key):
    return itertools.groupby(sorted(tuples_iterable, key=key), key)

# Iterates through a list having the previous and next items present
# Stolen from http://stackoverflow.com/a/323910/684253
def neighborhood(iterable):
    iterator = iter(iterable)
    prev = None
    item = iterator.next()  # throws StopIteration if empty.
    for next in iterator:
        yield (prev,item,next)
        prev = item
        item = next
    yield (prev,item,None)