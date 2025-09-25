import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class EmailCleanupService {
  private readonly logger = new Logger(EmailCleanupService.name);

  constructor(private readonly prisma: PrismaService) {}

  @Cron(CronExpression.EVERY_WEEK)
  async cleanupOldActions() {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);

    const deleted = await this.prisma.emailAction.deleteMany({
      where: { lastSentAt: { lt: oneHourAgo } },
    });
  }
}
