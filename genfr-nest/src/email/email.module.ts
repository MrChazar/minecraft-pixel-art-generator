import { Module } from '@nestjs/common';
import { EmailService } from './email.service';
import { join } from 'node:path'; 
import { MailerModule } from '@nestjs-modules/mailer';
import { HandlebarsAdapter } from '@nestjs-modules/mailer/dist/adapters/handlebars.adapter';
import { PrismaModule } from '../prisma/prisma.module';
import { EmailCleanupService } from './email-cleanup.service';

@Module({
  providers: [EmailService, EmailCleanupService],
  exports: [EmailService],
  imports: [MailerModule.forRoot({
      transport: {
        host: process.env.MAIL_HOST,
        port: process.env.MAIL_PORT,
        auth: {
          user: process.env.MAIL_USER,
          pass: process.env.MAIL_PASS,
        },
      },
      defaults: {
        from: '"Generator Frajdy" <${process.env.MAIL_USER}>',
      },
      template: {
        dir: process.env.NODE_ENV === 'production' 
          ? join(process.cwd(), 'dist/src/email/templates')
          : join(process.cwd(), 'src/email/templates'),
        adapter: new HandlebarsAdapter(),
        options: {
          strict: true,
        },
      },
    }),
    PrismaModule
  ]
})
export class EmailModule {}
